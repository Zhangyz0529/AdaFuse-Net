import pickle

import numpy as np
import torch




SPLIT_ALIASES = {
    None: None,
    "train": "meta_train",
    "val": "meta_val",
    "valid": "meta_val",
    "validation": "meta_val",
    "test": "meta_test",
    "meta_train": "meta_train",
    "meta_val": "meta_val",
    "meta_test": "meta_test",
}


class LOGODataManager:
    """Leakage-controlled multimodal dataset manager.

    Splits are created before normalization. Z-score statistics are fitted only
    on meta-train indices, then reused for meta-val and meta-test.
    """

    def __init__(self, data_pkl_path, cfg, seed=None, split_protocol=None):
        self.cfg = dict(cfg)
        self.data_pkl_path = data_pkl_path
        self.seed = int(self.cfg.get("seed", 0) if seed is None else seed)
        self.rng = np.random.default_rng(self.seed)
        self.split_protocol = split_protocol or self.cfg.get("split_protocol", "class_disjoint")

        print(f"Loading full dataset from {data_pkl_path}...")
        with open(data_pkl_path, "rb") as f:
            data = pickle.load(f)

        raw_stats, seqs, labels = self._extract_samples(data)
        raw_stats, seqs, labels, skipped = self._clean_samples(raw_stats, seqs, labels)
        if skipped > 0:
            print(f"Filtered {skipped} samples with invalid statistical features.")

        labels = np.array([self._map_label(label) for label in labels])
        seqs = np.asarray(seqs)
        raw_stats = np.asarray(raw_stats, dtype=np.float32)
        raw_stats, seqs, labels = self._limit_samples_per_class(raw_stats, seqs, labels)

        self.raw_stats = raw_stats
        self.seqs = seqs
        self.labels = labels
        self.indices_by_class = self._index_by_class(np.arange(len(self.labels)))
        self.class_rounds = self._build_class_splits()
        self.round_count = len(self.class_rounds)
        self.class_splits = self.class_rounds[0]
        self.split_indices_by_round = [
            self._build_index_splits_for_round(round_split)
            for round_split in self.class_rounds
        ]
        self.split_indices = self.split_indices_by_round[0]
        self.indices_by_split_by_round = [
            {
                split: self._index_by_class(indices)
                for split, indices in round_indices.items()
            }
            for round_indices in self.split_indices_by_round
        ]
        self.indices_by_split = self.indices_by_split_by_round[0]

        train_indices = self.split_indices["meta_train"]
        if len(train_indices) == 0:
            raise ValueError("meta_train split is empty; cannot fit normalization statistics.")
        self.mean = np.mean(self.raw_stats[train_indices], axis=0)
        self.std = np.std(self.raw_stats[train_indices], axis=0) + 1e-8
        self.stats = (self.raw_stats - self.mean) / self.std

        self._print_summary()

    def _extract_samples(self, data):
        raw_stats, seqs, labels = [], [], []
        for sample in data:
            stat = sample.get("stat_raw", sample.get("stat"))
            seq = sample.get("seq_feature", sample.get("seq"))
            label = sample.get("label")
            if stat is not None and seq is not None and label is not None:
                raw_stats.append(stat)
                seqs.append(seq)
                labels.append(str(label))
        return raw_stats, seqs, labels

    def _clean_samples(self, raw_stats, seqs, labels):
        clean_stats, clean_seqs, clean_labels = [], [], []
        skipped = 0
        for stat, seq, label in zip(raw_stats, seqs, labels):
            try:
                clean_stat = np.asarray(stat, dtype=np.float32).reshape(-1)
            except (TypeError, ValueError):
                try:
                    clean_stat = np.asarray(stat[7:-1], dtype=np.float32).reshape(-1)
                except (TypeError, ValueError):
                    skipped += 1
                    continue
            clean_stat = np.nan_to_num(clean_stat, nan=0.0, posinf=0.0, neginf=0.0)
            clean_stats.append(clean_stat)
            clean_seqs.append(seq)
            clean_labels.append(str(label))
        return clean_stats, clean_seqs, clean_labels, skipped

    def _map_label(self, label):
        label = str(label)
        if label in self.cfg.get("benign_mapping", []):
            return str(self.cfg["normal_label"])
        return str(self.cfg.get("attack_mapping", {}).get(label, label))

    def _limit_samples_per_class(self, raw_stats, seqs, labels):
        limit = self.cfg.get("max_samples")
        if limit is None:
            return raw_stats, seqs, labels

        keep_indices = []
        for cls in np.unique(labels):
            cls_indices = np.where(labels == cls)[0]
            if len(cls_indices) > limit:
                cls_indices = self.rng.choice(cls_indices, size=limit, replace=False)
            keep_indices.extend(cls_indices.tolist())

        keep_indices = np.array(sorted(keep_indices), dtype=int)
        print(f"Applied per-class cap max_samples={limit}; kept {len(keep_indices)} samples.")
        return raw_stats[keep_indices], seqs[keep_indices], labels[keep_indices]

    def _index_by_class(self, indices):
        return {
            cls: indices[self.labels[indices] == cls]
            for cls in np.unique(self.labels[indices])
        }

    def _configured_class_list(self, key):
        classes = self.cfg.get(key, [])
        if isinstance(classes, str):
            classes = [x.strip() for x in classes.split(",") if x.strip()]
        return [str(x) for x in classes]

    def _auto_zero_day_split(self, attack_classes):
        if len(attack_classes) == 0:
            return [], []
        if len(attack_classes) == 1:
            return [], attack_classes

        shuffled = list(attack_classes)
        self.rng.shuffle(shuffled)
        if len(shuffled) == 2:
            return [shuffled[0]], [shuffled[1]]

        n_val = max(1, int(round(len(shuffled) * self.cfg.get("val_class_ratio", 0.2))))
        n_test = max(1, int(round(len(shuffled) * self.cfg.get("test_class_ratio", 0.2))))
        if n_val + n_test >= len(shuffled):
            n_val = 1
            n_test = 1

        test_classes = shuffled[-n_test:]
        val_classes = shuffled[-(n_test + n_val):-n_test]
        return val_classes, test_classes

    def _build_class_splits(self):
        normal_label = str(self.cfg.get("normal_label"))
        all_classes = [str(c) for c in np.unique(self.labels)]
        configured_attacks = [str(c) for c in self.cfg.get("attack_classes", [])]
        attack_classes = [c for c in configured_attacks if c in all_classes and c != normal_label]

        if self.split_protocol == "class_rotate":
            return self._build_rotation_rounds(attack_classes, all_classes, normal_label)

        val_attacks = [c for c in self._configured_class_list("val_attack_classes") if c in attack_classes]
        test_attacks = [
            c for c in self._configured_class_list("test_attack_classes")
            if c in attack_classes and c not in val_attacks
        ]
        if not val_attacks and not test_attacks:
            val_attacks, test_attacks = self._auto_zero_day_split(attack_classes)

        held_out = set(val_attacks) | set(test_attacks)
        seen_attacks = [c for c in attack_classes if c not in held_out]
        other_seen = [
            c for c in all_classes
            if c != normal_label and c not in attack_classes and c not in held_out
        ]

        def with_normal(classes):
            if normal_label in all_classes:
                return [normal_label] + list(classes)
            return list(classes)

        return [{
            "meta_train": with_normal(seen_attacks + other_seen),
            "meta_val": with_normal(val_attacks),
            "meta_test": with_normal(test_attacks),
            "val_attack_classes": val_attacks,
            "test_attack_classes": test_attacks,
        }]

    def _build_rotation_rounds(self, attack_classes, all_classes, normal_label):
        def with_normal(classes):
            if normal_label in all_classes:
                return [normal_label] + list(classes)
            return list(classes)

        other_seen = [
            c for c in all_classes
            if c != normal_label and c not in attack_classes
        ]
        rounds = []
        total_attacks = len(attack_classes)
        val_novel = min(int(self.cfg.get("round_val_novel_classes", 1)), max(0, total_attacks - 1))
        test_novel = min(int(self.cfg.get("novel_classes_per_round", 1)), max(1, total_attacks))

        if total_attacks == 0:
            rounds.append({
                "meta_train": with_normal(other_seen),
                "meta_val": with_normal([]),
                "meta_test": with_normal([]),
                "val_attack_classes": [],
                "test_attack_classes": [],
            })
            return rounds

        for start in range(total_attacks):
            test_attacks = [attack_classes[(start + i) % total_attacks] for i in range(test_novel)]
            val_attacks = [attack_classes[(start + test_novel + i) % total_attacks] for i in range(val_novel)]
            held_out = set(test_attacks) | set(val_attacks)
            seen_attacks = [c for c in attack_classes if c not in held_out]

            rounds.append({
                "meta_train": with_normal(seen_attacks + other_seen),
                "meta_val": with_normal(val_attacks),
                "meta_test": with_normal(test_attacks),
                "val_attack_classes": val_attacks,
                "test_attack_classes": test_attacks,
            })

        return rounds

    def _build_index_splits_for_round(self, round_split):
        split_indices = {"meta_train": [], "meta_val": [], "meta_test": []}
        normal_label = str(self.cfg.get("normal_label"))
        val_attacks = set(round_split["val_attack_classes"])
        test_attacks = set(round_split["test_attack_classes"])
        train_ratio, val_ratio, _ = self.cfg.get("normal_split_ratios", (0.6, 0.2, 0.2))

        for cls, indices in self.indices_by_class.items():
            shuffled = self.rng.permutation(indices)
            if cls == normal_label:
                train_end = int(len(shuffled) * train_ratio)
                val_end = train_end + int(len(shuffled) * val_ratio)
                if train_end == 0 and len(shuffled) > 0:
                    train_end = 1
                split_indices["meta_train"].extend(shuffled[:train_end])
                split_indices["meta_val"].extend(shuffled[train_end:val_end])
                split_indices["meta_test"].extend(shuffled[val_end:])
            elif cls in val_attacks:
                split_indices["meta_val"].extend(shuffled)
            elif cls in test_attacks:
                split_indices["meta_test"].extend(shuffled)
            else:
                split_indices["meta_train"].extend(shuffled)

        return {
            split: np.array(sorted(indices), dtype=int)
            for split, indices in split_indices.items()
        }

    def _normalize_split(self, split):
        split = SPLIT_ALIASES.get(split, split)
        if split is None:
            return None
        if split not in self.indices_by_split:
            raise ValueError(f"Unknown split '{split}'.")
        return split

    def _normalize_round(self, round_index):
        if round_index is None:
            return 0
        round_index = int(round_index)
        if round_index < 0 or round_index >= self.round_count:
            raise ValueError(f"round_index must be between 0 and {self.round_count - 1}, got {round_index}.")
        return round_index

    def get_classes(self, split=None, round_index=None):
        split = self._normalize_split(split)
        if split is None:
            return list(self.indices_by_class.keys())
        round_index = self._normalize_round(round_index)
        return [
            cls for cls in self.class_rounds[round_index][split]
            if len(self.indices_by_split_by_round[round_index][split].get(cls, [])) > 0
        ]

    def get_indices_for_class(self, cls, split=None, round_index=None):
        split = self._normalize_split(split)
        if split is None:
            return self.indices_by_class.get(cls, np.array([], dtype=int))
        round_index = self._normalize_round(round_index)
        return self.indices_by_split_by_round[round_index][split].get(cls, np.array([], dtype=int))

    def _print_summary(self):
        unique, counts = np.unique(self.labels, return_counts=True)
        print("Dataset label distribution:")
        for cls, count in zip(unique, counts):
            print(f"  - [{cls:<20}]: {count:>8} samples")

        print("Leakage-controlled protocol:")
        print(f"  split_protocol = {self.split_protocol}")
        print(f"  seed = {self.seed}")
        print(f"  rotation rounds = {self.round_count}")
        for split in ["meta_train", "meta_val", "meta_test"]:
            classes = self.get_classes(split)
            n_samples = len(self.split_indices[split])
            print(f"  {split}: {len(classes)} classes, {n_samples} samples, classes={classes}")
        if self.split_protocol == "class_rotate":
            for ridx in range(self.round_count):
                round_split = self.class_rounds[ridx]
                print(f"    round {ridx}: val={round_split['val_attack_classes']} test={round_split['test_attack_classes']}")
        print("  normalization = fitted on meta_train only")


class EpisodeGenerator:
    def __init__(self, dataset, classes, split=None, round_index=None, seed=None, strict=True, min_query=None):
        self.dataset = dataset
        self.split = SPLIT_ALIASES.get(split, split)
        self.round_index = round_index
        self.strict = strict
        self.cfg = dict(dataset.cfg)
        self.min_query = int(self.cfg.get("min_query_num", 1) if min_query is None else min_query)
        self.rng = np.random.default_rng(self.cfg.get("seed", 0) if seed is None else seed)

        self.classes = []
        self.cat_idx = {}
        for cls in classes:
            indices = dataset.get_indices_for_class(cls, self.split, round_index=self.round_index)
            if len(indices) > 0:
                self.classes.append(cls)
                self.cat_idx[cls] = indices

        if len(self.classes) == 0:
            raise ValueError(f"No available classes for split={self.split}.")

    def eligible_classes(self, k_shot, q_query):
        return [
            cls for cls in self.classes
            if len(self.cat_idx[cls]) >= int(k_shot) + self.min_query
        ]

    def get_episode(self, n_way, k_shot, q_query, return_indices=False):
        eligible = self.eligible_classes(k_shot, q_query)
        curr_way = min(int(n_way), len(eligible))
        if curr_way < 2:
            raise ValueError(
                f"Need at least 2 eligible classes, got {curr_way}. "
                f"k_shot={k_shot}, q_query={q_query}, split={self.split}"
            )

        selected_classes = self.rng.choice(eligible, curr_way, replace=False)
        s_stat, s_seq, q_stat, q_seq = [], [], [], []
        support_indices, query_indices, query_labels = [], [], []

        for label_id, cls in enumerate(selected_classes):
            pool = self.cat_idx[cls]
            actual_query = min(int(q_query), len(pool) - int(k_shot))
            if actual_query < self.min_query:
                if self.strict:
                    raise ValueError(
                        f"Class {cls} has only {len(pool)} samples; "
                        f"cannot build {k_shot}-shot episode with query samples."
                    )
                continue

            chosen = self.rng.choice(pool, int(k_shot) + actual_query, replace=False)
            s_idx = chosen[:int(k_shot)]
            q_idx = chosen[int(k_shot):]

            s_stat.append(torch.as_tensor(self.dataset.stats[s_idx], dtype=torch.float32))
            s_seq.append(torch.as_tensor(self.dataset.seqs[s_idx], dtype=torch.long))
            q_stat.append(torch.as_tensor(self.dataset.stats[q_idx], dtype=torch.float32))
            q_seq.append(torch.as_tensor(self.dataset.seqs[q_idx], dtype=torch.long))
            support_indices.extend(s_idx.tolist())
            query_indices.extend(q_idx.tolist())
            query_labels.extend([label_id] * len(q_idx))

        if len(s_stat) != curr_way:
            raise ValueError("Episode construction failed because too few classes met the sampling constraints.")

        episode = (
            torch.stack(s_stat).reshape(-1, s_stat[0].shape[-1]),
            torch.stack(s_seq).reshape(-1, s_seq[0].shape[-1]),
            torch.cat(q_stat, dim=0),
            torch.cat(q_seq, dim=0),
            torch.as_tensor(query_labels, dtype=torch.long),
        )

        if return_indices:
            audit = {
                "classes": [str(c) for c in selected_classes],
                "support_indices": support_indices,
                "query_indices": query_indices,
                "overlap": sorted(set(support_indices) & set(query_indices)),
            }
            return episode + (audit,)

        return episode
