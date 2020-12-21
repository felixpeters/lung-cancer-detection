import pylidc as pl
scans = pl.query(pl.Scan)
print(f"Total scans in dataset: {scans.count()}")
pid = 'LIDC-IDRI-0078'
patient_scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid)
print(f"Total scans for patient 78: {patient_scans.count()}")
scan = patient_scans.first()
print(f"Total annotations in first scan: {len(scan.annotations)}")
nods = scan.cluster_annotations()
print(f"Total nodules found in first scan: {len(nods)}")
vol = scan.to_volume()
print(f"Dimensions of first scan: {vol.shape}")
scan.visualize(annotation_groups=nods)
