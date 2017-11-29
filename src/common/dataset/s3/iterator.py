
# Based off -
# https://stackoverflow.com/questions/31918960/boto3-to-download-all-files-from-a-s3-bucket


def s3_iterator(client, resource, root, dir, bucket, action):
    paginator = client.get_paginator('list_objects')

    for result in paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=dir):
        if result.get('CommonPrefixes') is not None:
            for subdir in result.get('CommonPrefixes'):
                s3_iterator(client, resource, root, subdir.get('Prefix'), bucket, action)
        if result.get('Contents') is not None:
            for file in result.get('Contents'):
                action(file.get("Key").replace(root,""))


                #print(file.get('Key').replace(dist,""))

                #obj = client.get_object(Bucket=bucket, Key=file.get("Key"))
                #writer.save(file.get("Key").replace(dist,""), obj["Body"].read().decode("utf-8"))

                #resource.meta.client.download_file(bucket, file.get('Key'), local + os.sep + clean(file.get('Key')))
