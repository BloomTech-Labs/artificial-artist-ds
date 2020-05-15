import os

S3_BUCKET                 = os.environ.get("S3_BUCKET")
S3_KEY                    = os.environ.get("S3_KEY")
S3_SECRET                 = os.environ.get("S3_SECRET_ACCESS_KEY")
S3_LOCATION               = 'https://{}.s3.amazonaws.com/'.format(S3_BUCKET)

SECRET_KEY                = os.urandom(32)
PORT                      = 5000