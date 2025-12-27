# File Record in MongoDB structure

## File Record structure

###    _id
    sample: 321 
description: Unique identifier for the file record

### name
    "file_name.jpg"
description:

###    local_path
    "path/to/file"
description:

###    source_uri
    "http://path/to/file"
description:

### extension
    "jpg"
description: file extension

### size
    123456

### file_type
    "image"
description:

    "created": "2021-01-01T00:00:00",
    "modified": "2021-01-01T00:00:00",
    "file_md5": "1234567890abcdef",
    "file_content_md5": "1234567890abcdef",
    "tags": [321, 321], //reference to tags
    "categories": [321, 321], //reference to categories
    "title": "This is a title of the file",
    "description": "This is a description of the file",
    "note": "This is a note for the file", 
    "rating": 5, //rating from 1 to 5'
    "favorite": true, //favorite status
    "embeddings": [321, 321], //reference to embedings
    "thumbnails": [
      {
        "size": "256x256",
        "path": "path/to/thumbnail"
      },
      {
        "size": "512x512",
        "path": "path/to/thumbnail"
      }],
    "annotations": [123, 123], //reference to annotations
    "indexed_by": ["metadata_indexer","tensor_mobile_net"], //indexer list
    "metadata": {
      "dirty": false, //dirty flag
      "modified": "2021-01-01T00:00:00",
  
    #, for image files,
    
        "image_read_error": false, //error flag
        "image_truncated": false, //truncated flag
        "width": 1920,
        "height": 1080,
        "dpi": 300,
        "color": "RGB"
    },
      "external_metadata": {
        "source": "source_name",
        "source_id": "source_id",
        "source_metadata": {
          "key1": "value1",
          "key2": "value2"
        }
      }



### File types:
- image
- video
- audio
- document
- other

# Tag record in MongoDB

## Tag Record

### _id
    : 321,

### name
deacription  "tag_name"

### description
    "This is a description of the tag"

# parent_tag": 123, //reference to parent tag
    "AI": false
### path

### sinonims
