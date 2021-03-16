import requests
from requests.auth import HTTPBasicAuth
import pydicom


#orthanc = Orthanc('https://orthanc.dicom.org.tw')
#orthanc.setup_credentials('cyorthanc', 'cy123orthanc')

LOGIN_URL = "https://orthanc.dicom.org.tw/wado/?requestType=WADO&studyUID=1.2.840.113817.20170608.410950419.1213115.52&seriesUID=1.2.840.113564.102241028.2017060816251139053&objectUID=1.2.840.113564.102241028.2017060816251140655.1003000225002&contentType=application%2Fdicom"
session_requests = requests.session()
r = requests.get(LOGIN_URL,auth=HTTPBasicAuth('cyorthanc', 'cy123orthanc'))
#print(r.content)

with open("2.dcm", "wb") as code:
    code.write(r.content)
    pass


data=pydicom.dcmread("2.dcm")

print(data.PatientName)
