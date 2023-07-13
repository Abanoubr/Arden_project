credentials = {
	"COS": {
		"apikey": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
		"endpoint": "https://s3.eu-de.cloud-object-storage.appdomain.cloud",
		"resource_instance_id": "crn:v1:bluemix:public:cloud-object-storage:global:a/64ea99716bb528463c1a86403efa2208:bf541bfa-9570-423b-901d-019da38a0fb8::"
	},
	"WATSON": {
		"DEV_1": {
			"instance": "https://api.eu-de.assistant.watson.cloud.ibm.com/instances/33b789c5-54de-4b0d-85cb-5cc44f15c590",
			"api_key": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
		},
		"DEV_2": {
			"instance": "https://api.eu-de.assistant.watson.cloud.ibm.com/instances/2178a839-23e6-4ed2-99e8-c6a87eb05611",
			"api_key": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
		},
		"TEST_1": {
			"instance": "https://api.eu-de.assistant.watson.cloud.ibm.com/instances/b1ce2268-82ae-445f-837f-ff636308f48a",
			"api_key": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
		},
		"TEST_2": {
			"instance": "https://api.eu-de.assistant.watson.cloud.ibm.com/instances/757932b8-8b16-43a5-ba9c-5a59b7c10c79",
			"api_key": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
		}
	}
}

def get_credentials(service):
	return credentials[service]