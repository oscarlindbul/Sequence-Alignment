from qiskit import IBMQ

# load API
cred_file = open("creds.txt")
creds = cred_file.read()
print("API:", creds)
IBMQ.save_account(creds, overwrite=True)
