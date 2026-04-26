import pinecone
from config import settings

pc = pinecone.Pinecone(api_key=settings.pinecone_api_key)
index = pc.Index(settings.pinecone_index_name)

print(f"Connected to index: {settings.pinecone_index_name}")

for ids in index.list():
    fetch_response = index.fetch(ids=ids)

    for vec_id, data in fetch_response['vectors'].items():
        old_path = data.metadata.get('image_path', '')

        if old_path.startswith('data/'):
            new_path = old_path.replace('data/', 'images/', 1)
            index.update(
                id=vec_id,
                set_metadata={"image_path": new_path}
            )
            print(f"Updated path for ID: {vec_id}")

print("Metadata update complete.")
