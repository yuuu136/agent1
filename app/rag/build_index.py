from app.rag.service import rag_service


def main() -> None:
    result = rag_service.build_index()
    print(result)


if __name__ == "__main__":
    main()
