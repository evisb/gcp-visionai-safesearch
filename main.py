import os
from csv import reader, writer
import requests
from bs4 import BeautifulSoup
from google.cloud import bigquery as bq
from google.cloud import storage as gcs
from google.cloud import vision


# Client object for the BigQuery service.
bq_client = bq.Client()

# Client object for the Google Cloud Storage service.
gcs_client = gcs.Client()

# Client object for the Google Vision API.
vision_client = vision.ImageAnnotatorClient()

# Bucket for the images.
bucket_name = 'eximapiamtmlbckt01'
bucket = gcs_client.create_bucket(bucket_name)
print('Bucket {} created'.format(bucket.name))

# A table for storing the URLs and the scoring information.
dataset_id = 'images'
dataset_ref = bq.Dataset(bq_client.project + '.' + dataset_id)
dataset = bq_client.create_dataset(dataset_ref)
print('Created dataset {}.{}'.format(bq_client.project, dataset.dataset_id))

table_id = bq_client.dataset(dataset_id).table('images')
schema = [
    bq.SchemaField('image_id', 'STRING', mode='REQUIRED'),
    bq.SchemaField('image_url', 'STRING', mode='REQUIRED'),
    bq.SchemaField('previously_rejected', 'INTEGER', mode='REQUIRED'),
    bq.SchemaField('vai_adult_score', 'STRING', mode='REQUIRED'),
    bq.SchemaField('vai_medical_score', 'STRING', mode='REQUIRED'),
    bq.SchemaField('vai_violence_score', 'STRING', mode='REQUIRED'),
    bq.SchemaField('vai_racy_score', 'STRING', mode='REQUIRED'),
    bq.SchemaField('vai_spoof_score', 'STRING', mode='REQUIRED'),
    bq.SchemaField('should_be_rejected', 'INTEGER', mode='REQUIRED'),
]

table_ref = bq.Table(table_id, schema=schema)
table = bq_client.create_table(table_ref)
print('Created table {}.{}.{}'.format(table.project, table.dataset_id, table.table_id))

# Optionally obtain the image URLs from the website if not already available.
url = 'https://www.google.com/search?q=[SEARCH_TERM]&tbm=isch'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
images = soup.find_all('img')
image_urls = [image['src'] for image in images]
print('Found {} images'.format(len(image_urls) - 1))

# Create a file for the image URLs.
# Store the images to the GCP bucket and write each object URL to the file together
# with their previously_rejected value defaulting to 0.
images_file = 'images.csv'
images_file_path = os.path.join(os.path.dirname(__file__), images_file)

with open(images_file_path, 'a') as f:
    for image_url in image_urls:
        image_id = image_url.split('/')[-1]
        if image_url.startswith('https'):
            blob = bucket.blob(image_id)
            blob.upload_from_string(requests.get(image_url).content)
            writer(f).writerow([image_id, "gs://" + bucket.name + "/" + blob.name, 0])
        else:
            del image_urls[image_urls.index(image_url)]

print('Stored {} images to bucket {}'.format(len(image_urls) - 1, bucket.name))

# Load the data from the file into memory and call the Google Vision API to get the scores for each image.
# Store the image_id, image URL, previously_rejected, should_be_rejected and their scores in the scores
# output file.
scores_file = 'scores.csv'
scores_file_path = os.path.join(os.path.dirname(__file__), scores_file)

likelihood = {
    'UNKNOWN': 0,
    'VERY_UNLIKELY': 1,
    'UNLIKELY': 2,
    'POSSIBLE': 3,
    'LIKELY': 4,
    'VERY_LIKELY': 5
}

with open(images_file_path, 'r') as f, open(scores_file_path, 'a') as k:
    csv_reader = reader(f)
    csv_writer = writer(k)
    for row in csv_reader:
        image_id = row[0]
        image_url = row[1]
        previously_rejected = row[2]
        response = vision_client.annotate_image({
            'image': {'source': {'image_uri': image_url}},
            'features': [{'type_': vision.Feature.Type.SAFE_SEARCH_DETECTION}],
        })
        vai_adult_score = list(likelihood.values())[response.safe_search_annotation.adult]
        vai_medical_score = list(likelihood.values())[response.safe_search_annotation.medical]
        vai_violence_score = list(likelihood.values())[response.safe_search_annotation.violence]
        vai_racy_score = list(likelihood.values())[response.safe_search_annotation.racy]
        vai_spoof_score = list(likelihood.values())[response.safe_search_annotation.spoof]
        should_be_rejected = 1 if vai_adult_score >= likelihood['LIKELY'] or \
            vai_medical_score >= likelihood['LIKELY'] or \
            vai_violence_score >= likelihood['LIKELY'] or \
            vai_racy_score >= likelihood['LIKELY'] or \
            vai_spoof_score >= likelihood['LIKELY'] else 0
        csv_writer.writerow([image_id, image_url, previously_rejected, vai_adult_score, \
            vai_medical_score, vai_violence_score, vai_racy_score, vai_spoof_score, should_be_rejected])


# Load the data from the file into the table.
with open(scores_file_path, 'r') as f:
    reader = reader(f)
    rows_to_insert = []
    for row in reader:
        rows_to_insert.append({
            'image_id': row[0],
            'image_url': row[1],
            'previously_rejected': row[2],
            'vai_adult_score': row[3],
            'vai_medical_score': row[4],
            'vai_violence_score': row[5],
            'vai_racy_score': row[6],
            'vai_spoof_score': row[7],
            'should_be_rejected': row[8],
        })
    errors = bq_client.insert_rows(table, rows_to_insert)
    assert errors == []
print('Loaded {} rows into table {}.{}.{}'.format(len(rows_to_insert), table.project, table.dataset_id, table.table_id))
