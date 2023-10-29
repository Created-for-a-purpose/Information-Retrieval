from elasticsearch import Elasticsearch

# Initialize the Elasticsearch client with the scheme specified
es = Elasticsearch(scheme='http', hosts=[{'host': 'localhost', 'port': 9200}])  # Replace with your Elasticsearch server details


index_name = 'newsgroups_index'  # Replace with your index name

# Example: Retrieve a document by its ID
document_id = 1
document = es.get(index=index_name, id=document_id)
text_to_vectorize = document['_source']['document_text']
