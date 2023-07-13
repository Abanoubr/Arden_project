import ibm_boto3
from ibm_botocore.client import Config, ClientError
from controller import credentials

# Source credentials from file
creds = credentials.get_credentials("COS")

# Extract API Details
COS_API_KEY = creds["apikey"]
COS_ENDPOINT = creds["endpoint"]
COS_INSTANCE_CRN = creds["resource_instance_id"]

# Create client 
cos = ibm_boto3.resource("s3",
	ibm_api_key_id = COS_API_KEY,
	ibm_service_instance_id = COS_INSTANCE_CRN,
	config = Config(signature_version = "oauth"),
	endpoint_url = COS_ENDPOINT
)

# Test connection by fetching bucket list
def get_buckets():
	print("\nRetrieving Buckets List...\n")
	try:
		buckets = cos.buckets.all()
		print([i.name for i in buckets])
		print('Connection to IBM COS Successful')
	except ClientError as e:
		print("\nClient Error: {}\n".format(e))
	except Exception as e:
		print("\nUnable to retrieve bucket list: {}".format(e))

# Create item as text file
def create_text_file(bucket_name, item_name, file_path):
	print("Creating Remote Item: {0}".format(item_name))
	try:
		with open(file_path, 'r', encoding='UTF-8') as file:
			cos.Object(bucket_name, item_name).put(Body=file.read())
			print("Item: {0} Upload Successful\n".format(item_name))
	except ClientError as be:
		print("CLIENT ERROR: {0}\n".format(be))
	except Exception as e:
		print("Unable to create text file: {0}".format(e))
