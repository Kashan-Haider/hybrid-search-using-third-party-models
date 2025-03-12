from pinecone_text.sparse import BM25Encoder
from pinecone.grpc import PineconeGRPC as Pinecone
import os
from dotenv import load_dotenv
from pinecone import ServerlessSpec

load_dotenv()
pinecone_api = os.getenv("PINECON_API")
pc = Pinecone(api_key=pinecone_api)


essays = [
    {
        "id": 1,
        "title": "The Impact of Technology on Society",
        "content": "Technology has profoundly transformed modern society in various aspects. From communication and education to healthcare and industry, technological advancements have streamlined processes, improved efficiency, and enhanced the quality of life. However, technology has also raised concerns related to privacy, cybersecurity, and societal dependence on digital tools. Striking a balance between leveraging technology for progress and addressing its challenges remains a priority.",
    },
    {
        "id": 2,
        "title": "Climate Change and Its Global Effects",
        "content": "Climate change poses a significant threat to the environment and human life. The increase in global temperatures, rising sea levels, and frequent natural disasters are all consequences of climate change. Mitigation efforts, such as reducing carbon emissions, promoting sustainable practices, and adopting renewable energy sources, are crucial to curbing its effects and ensuring a safer planet for future generations.",
    },
    {
        "id": 3,
        "title": "The Importance of Mental Health Awareness",
        "content": "Mental health is as important as physical health, yet it remains stigmatized in many societies. Raising awareness about mental health issues can help individuals seek timely support and reduce the stigma associated with mental illness. Ensuring accessible mental health care and promoting well-being are essential for building a healthy and inclusive society.",
    },
    {
        "id": 4,
        "title": "The Future of Artificial Intelligence",
        "content": "Artificial Intelligence (AI) is rapidly advancing, reshaping industries and daily life. From autonomous vehicles to personalized healthcare, AI's potential is immense. However, ethical considerations, including job displacement, privacy concerns, and bias in algorithms, need to be addressed to ensure AI serves humanity positively.",
    },
    {
        "id": 5,
        "title": "The Role of Education in Personal Growth",
        "content": "Education is a powerful tool that empowers individuals with knowledge, skills, and critical thinking. It plays a fundamental role in personal growth, career opportunities, and social development. Accessible and quality education should be a priority for all societies to promote equality and progress.",
    },
    {
        "id": 6,
        "title": "The Benefits of Physical Exercise",
        "content": "Regular physical exercise is essential for maintaining a healthy body and mind. It improves cardiovascular health, boosts mood, enhances cognitive function, and reduces the risk of various diseases. Incorporating exercise into daily routines is a proactive approach to enhancing overall well-being.",
    },
    {
        "id": 7,
        "title": "The Importance of Environmental Conservation",
        "content": "Preserving the natural environment is crucial for sustaining biodiversity, mitigating climate change, and ensuring future generations' survival. Conservation efforts, such as reforestation, wildlife protection, and reducing plastic pollution, are vital for promoting ecological balance.",
    },
    {
        "id": 8,
        "title": "The Influence of Social Media on Youth",
        "content": "Social media has revolutionized how people interact and communicate. While it offers opportunities for self-expression and connection, excessive use can harm mental health, self-esteem, and productivity. Encouraging responsible usage is essential for fostering a positive digital environment.",
    },
    {
        "id": 9,
        "title": "The Evolution of Work in the Digital Age",
        "content": "The digital age has transformed traditional work structures, promoting remote work, gig economy platforms, and automation. While these changes offer flexibility and efficiency, they also raise concerns about job security, work-life balance, and fair wages. Preparing for the future of work requires adapting to technological advancements and prioritizing employee well-being.",
    },
    {
        "id": 10,
        "title": "The Power of Positive Thinking",
        "content": "Positive thinking can significantly impact an individual's mental and physical health. Cultivating an optimistic mindset promotes resilience, reduces stress, and enhances overall well-being. By focusing on solutions rather than problems, individuals can lead more fulfilling lives.",
    },
    {
        "id": 11,
        "title": "The Benefits of Multilingualism",
        "content": "Learning multiple languages offers cognitive, social, and cultural benefits. It enhances memory, improves problem-solving skills, and opens up new opportunities for personal and professional growth. Embracing multilingualism fosters greater empathy and understanding among diverse communities.",
    },
    {
        "id": 12,
        "title": "The Ethics of Genetic Engineering",
        "content": "Genetic engineering holds the potential to cure genetic disorders and improve crop yields. However, ethical concerns arise regarding its impact on natural evolution, biodiversity, and societal inequalities. Establishing guidelines and ethical standards is essential to ensure responsible scientific advancement.",
    },
    {
        "id": 13,
        "title": "The Rise of Remote Work",
        "content": "Remote work has gained popularity due to technological advancements and the global pandemic. While it offers flexibility and work-life balance, it also poses challenges related to communication, collaboration, and employee engagement. Companies must adapt to this evolving work culture.",
    },
    {
        "id": 14,
        "title": "The Impact of Fast Fashion",
        "content": "The fast fashion industry offers affordable clothing but at a high environmental and ethical cost. Overproduction, textile waste, and exploitative labor practices are significant issues. Advocating for sustainable fashion can help reduce the industry's negative impact.",
    },
    {
        "id": 15,
        "title": "The Power of Storytelling",
        "content": "Storytelling is a powerful tool for communication, education, and preserving cultural heritage. It allows individuals to convey ideas, emotions, and experiences, creating connections between people. Harnessing storytelling effectively can inspire and bring about social change.",
    },
]

corpus = [essay["content"] for essay in essays]

# Initialize BM25 and fit the corpus
bm25 = BM25Encoder()
bm25.fit(corpus)


# Encode the documents
def get_sparse_embeddings(texts):
    embedding = [bm25.encode_documents(text['content']) for text in texts]
    return embedding

def get_dense_embeddings(texts):
    embeddings = [
        pc.inference.embed(
            model="multilingual-e5-large",
            inputs=text['content'],
            parameters={"input_type": "passage", "truncate": "END"},
        )
        for text in texts
    ]
    return embeddings

def get_query_sparse_embedding(query):
    embedding = bm25.encode_documents(query) 
    return embedding

def get_query_dense_embedding(query):
     embeddings = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=query,
            parameters={"input_type": "passage", "truncate": "END"},
        )
     return embeddings

index_name = "hybrid-search-using-third-party-models-index"

# Create index if it doesn't exist
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1024,  # Dimension of dense embeddings
        metric="dotproduct",  # Use dotproduct for similarity measurement
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        deletion_protection="disabled",
    )
    print("Successfully created index " + index_name)

index = pc.Index("hybrid-search-using-third-party-models-index")
batch_size = 5
for i in range(0, len(corpus), batch_size):
    batch = essays[i : i + batch_size]
    dense_embeds = get_dense_embeddings(batch)
    sparse_embeds = get_sparse_embeddings(batch)
    vectors = []
    for j, item in enumerate(batch):
        dense_values = dense_embeds[j].data[0]['values']  # Extract dense values
        sparse_indices = sparse_embeds[j]["indices"]
        sparse_values = sparse_embeds[j]["values"]
        vectors.append(
            {
                "id": str(item["id"]),
                "values": dense_values,
                "metadata": {"title": item["title"], "content": item['content']},
                "sparse_values": {"indices": sparse_indices, "values": sparse_values},
            }
        )
    upsert_response = index.upsert(vectors=vectors)
    
    
query = "Global warming is a major issue"

query_dense_embeddings = get_query_dense_embedding(query)
query_sparse_embeddings = get_query_sparse_embedding(query)

# print(query_sparse_embeddings['indices'])

query_response = index.query(
    top_k=3,
    vector=query_dense_embeddings.data[0]["values"],
     include_metadata=True,
    sparse_vector={
        'indices': query_sparse_embeddings['indices'],
        'values':  query_sparse_embeddings['values']
    }
)
for data in query_response.matches:
    print(f"score: {data['score']} --- Title: {data.metadata['title']}")