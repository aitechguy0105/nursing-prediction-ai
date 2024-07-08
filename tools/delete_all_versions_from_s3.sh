#!/bin/bash
# Use this script to delete all versions from a bucket that has Versioning Enabled
# NOTE: each time it deletes 1000 versions (which is max that delete-objects can take). So, run this multiple times till you
# delete all the files!

# Update the next 4 lines before running!!
bucket=saiva-prod-data-bucket
prefixToDelete=data/dycora
# get date of "7 days ago"
deleteBefore=`date -v-7d +%F`

fileName='aws_delete.json'
rm $fileName
versionsToDelete=`aws s3api list-object-versions --bucket "$bucket" --max-items 1000 --prefix "$prefixToDelete" --query "Versions[?(LastModified<'$deleteBefore')].{Key: Key, VersionId: VersionId}"`

cat << EOF > $fileName
{
  "Objects":$versionsToDelete, 
  "Quiet": true
}
EOF

# Review the file created above and then
# uncomment the next line to actually delete the files!
# aws s3api delete-objects --bucket "$bucket" --delete file://$fileName
