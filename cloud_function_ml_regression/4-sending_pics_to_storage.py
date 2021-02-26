#from google.cloud import storage


def upload_blob(source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    bucket_name = "--yourbucketname--"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "4 - GCPics -File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

upload_blob("../tmp/RFE_columns.png","RFE_columns.png")
upload_blob("../tmp/outliers_2d.png","outliers_2d.png")
upload_blob("../tmp/outliers_3d.png","outliers_23.png")
