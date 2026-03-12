import os
import time
import argparse
from dotenv import load_dotenv, set_key
from pinecone import Pinecone, ServerlessSpec

DEFAULT_INDEX_NAME = "steam-reviews"
DEFAULT_NAMESPACE = "steam-reviews"
EMBED_DIM = 1536
METRIC = "cosine"

def main():
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--env_path", default=".env")
    ap.add_argument("--index", default=os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME))
    ap.add_argument("--namespace", default=os.getenv("PINECONE_NAMESPACE", DEFAULT_NAMESPACE))
    ap.add_argument("--cloud", default="aws")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--print_host", action="store_true", help="Print only the index host and exit (for entrypoint.sh).")
    args = ap.parse_args()

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY must be set in .env or environment")

    pc = Pinecone(api_key=api_key)

    index_name = args.index

    # ----------------------------
    # Check if index exists
    # ----------------------------
    existing_indexes = [i["name"] if isinstance(i, dict) else i.name for i in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"Index '{index_name}' not found. Creating it...")

        pc.create_index(
            name=index_name,
            dimension=EMBED_DIM,
            metric=METRIC,
            spec=ServerlessSpec(
                cloud=args.cloud,
                region=args.region,
            ),
        )

        # Wait until index is ready
        while True:
            desc = pc.describe_index(index_name)
            status = desc.get("status", {}) if isinstance(desc, dict) else getattr(desc, "status", {})
            if status.get("ready", False):
                break
            print("Waiting for index to become ready...")
            time.sleep(3)

        print("Index created and ready.")
    else:
        print(f"Index '{index_name}' already exists.")

    # ----------------------------
    # Fetch host
    # ----------------------------
    desc = pc.describe_index(index_name)
    
    if isinstance(desc, dict):
        host = desc.get("host") or desc.get("status", {}).get("host")
    else:
        host = getattr(desc, "host", None) or getattr(desc.status, "host", None)

    if not host:
        raise ValueError("Could not determine Pinecone index host.")
    
    # If entrypoint wants just the host, print it and exit.
    if args.print_host:
        print(host)
        return

    # ----------------------------
    # Write .env
    # ----------------------------
    if not os.path.exists(args.env_path):
        open(args.env_path, "a", encoding="utf-8").close()

    set_key(args.env_path, "PINECONE_API_KEY", api_key)
    set_key(args.env_path, "PINECONE_INDEX_NAME", index_name)
    set_key(args.env_path, "PINECONE_INDEX_HOST", host)
    set_key(args.env_path, "PINECONE_NAMESPACE", args.namespace)

    print("\nPinecone setup complete.")
    print("Index name :", index_name)
    print("Host       :", host)
    print("Namespace  :", args.namespace)
    print(f"Written to : {args.env_path}")

if __name__ == "__main__":
    main()
