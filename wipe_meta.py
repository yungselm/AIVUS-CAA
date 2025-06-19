import pydicom as dcm
from pydicom.uid import generate_uid

# 1. Read the file
ds = dcm.dcmread("test_cases/patient_example/anonymized.dcm")
print(ds)
# # 2. Remove PHI tags instead of setting "Nan"
# for tag in [
#     "PatientName",      # (0010,0010) PN
#     "PatientID",        # (0010,0020) LO
#     "PatientBirthDate", # (0010,0030) DA
#     "PatientSex",       # (0010,0040) CS
#     "StudyID",          # (0020,0010) SH
#     "PerformingPhysicianName",  # (0008,1050) PN
#     "InstanceCreationDate",
#     "StudyDate",
#     "SeriesDate",
#     "AcquisitionDate",
#     "ContentDate",
#     "AcquisitionDateTime",
# ]:
#     if tag in ds:
#         delattr(ds, tag)

# # 3. If you still need those attributes defined, set them to the correct “empty” VR:
# #    - PN, LO, SH, and other string VRs can just be set to "", which is valid.
# #    - DA must be "" (empty) rather than "Nan".
# #    - CS likewise can be set to "" (meaning “unknown”).
# ds.PatientName = ""
# ds.PatientID   = ""
# ds.PatientBirthDate = ""   # valid DA
# ds.PatientSex  = ""         # valid CS (empty)
# ds.StudyID     = ""
# ds.PerformingPhysicianName = ""

# # 4. Remove any private tags
# ds.remove_private_tags()

# # 5. Fix the File Meta as before
# meta = ds.file_meta
# new_sop = generate_uid()
# meta.MediaStorageSOPInstanceUID = new_sop
# ds.SOPInstanceUID              = new_sop
# meta.ImplementationClassUID    = generate_uid()

# # 6. Write out
# ds.save_as("test_cases/patient_example/anonymized.dcm")
