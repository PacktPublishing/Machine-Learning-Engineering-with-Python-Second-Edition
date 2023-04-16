import boto3
import http.client
import base64
import ast


# mwaa_env_name = 'YOUR_ENVIRONMENT_NAME'
# dag_name = 'YOUR_DAG_NAME'


def trigger_dag(mwaa_env_name: str, dag_name: str) -> str:
    client = boto3.client('mwaa')
    
    # get web token
    mwaa_cli_token = client.create_cli_token(
        Name=mwaa_env_name
    )
    
    conn = http.client.HTTPSConnection(mwaa_cli_token['WebServerHostname'])
    mwaa_cli_command = 'dags trigger'
    payload = mwaa_cli_command + " " + dag_name
    headers = {
      'Authorization': 'Bearer ' + mwaa_cli_token['CliToken'],
      'Content-Type': 'text/plain'
    }
    conn.request("POST", "/aws_mwaa/cli", payload, headers)
    res = conn.getresponse()
    data = res.read()
    dict_str = data.decode("UTF-8")
    mydata = ast.literal_eval(dict_str)
    return base64.b64decode(mydata['stdout']).decode('ascii')
    

