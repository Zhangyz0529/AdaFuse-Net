"""Create separately named complete-record prefixes, never alter original captures."""
import argparse,hashlib,json,csv
from pathlib import Path
from audit_truncation_and_domains import boundary
from rules_v3 import stable
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);a=p.parse_args();rows=json.loads((HERE/'diagnostics/CIC_DDOS_TRUNCATION_FORENSICS.json').read_text());out=HERE/'sources/verified_record_prefixes';out.mkdir(parents=True,exist_ok=True);registry=list(csv.DictReader((HERE/'source_domain_registry.csv').open()));reports=[]
for r in rows:
 source=Path(r['source_file']);rel=str(source.relative_to(a.data_root));record=next(v for v in registry if v['source_relative_path']==rel);n=r['record_boundary']['complete_record_prefix_bytes'];dest=out/(record['source_file_id']+'.pcap');h=hashlib.sha256()
 with source.open('rb') as f,dest.open('wb') as g:
  remaining=n
  while remaining:
   b=f.read(min(4<<20,remaining))
   if not b:raise ValueError('Source shorter than verified prefix')
   g.write(b);h.update(b);remaining-=len(b)
 check=boundary(dest);assert check['status']=='COMPLETE'
 cutoff=min(r['record_boundary'].get('partial_record_timestamp',float('inf')),r['record_boundary']['last_complete_epoch'])
 reports.append(dict(source_file_id=record['source_file_id'],provenance_domain_id=record['provenance_domain_id'],original_relative_path=rel,original_sha256=r['local']['sha256'],working_prefix=str(dest.relative_to(HERE)),working_prefix_sha256=h.hexdigest(),retained_complete_record_bytes=n,excluded_tail_bytes=r['local']['bytes']-n,excluded_partial_packet_count=1,source_archived_member_is_also_truncated=r['conclusion']=='LOCAL_ARCHIVE_MEMBER_ALSO_TRUNCATED',available_time_cutoff_epoch=cutoff,eligibility='reject CSV records whose calibrated start-time uncertainty plus duration reaches cutoff; no full-capture completeness claim',status='VERIFIED_PREFIX_FOR_SOURCE_RANGE_REVIEW_NOT_FORMAL_DATA',original_file_unchanged=True))
(HERE/'sources/PREFIX_PROVENANCE.json').write_text(json.dumps(reports,indent=2));print([{k:r[k] for k in ['original_relative_path','retained_complete_record_bytes','excluded_tail_bytes','available_time_cutoff_epoch']} for r in reports])
