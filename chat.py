from pathlib import Path

from core.knowledge import KnowledgeBase
from core.memory import ConversationMemory


BASE_DIR = Path(__file__).resolve().parent

knowledge = KnowledgeBase(
    str(BASE_DIR / "data" / "knowledge.json")
)

memory = ConversationMemory(
    str(BASE_DIR / "data" / "conversations.db")
)


def show_history():
    conversations = memory.recent(10)

    if not conversations:
        print("\nMyllers: Ainda não tenho conversas salvas.\n")
        return

    print("\n========== MEMÓRIA MYLLERS ==========")

    for question, answer in conversations:
        print(f"\nVocê: {question}")
        print(f"Myllers: {answer}")

    print("\n=====================================\n")


def main():
    print("=" * 42)
    print("          MYLLERS NEURAL v0.5")
    print("       Inteligência experimental")
    print("             Criador: Tallexs")
    print("=" * 42)

    print("\nSistema de conhecimento carregado.")
    print("Memória persistente ativada.")
    print("\nComandos: historico | sair\n")

    try:
        while True:
            try:
                question = input("Você: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nMyllers: Até a próxima!")
                break

            if not question:
                continue

            command = question.lower().strip()

            if command in ("sair", "exit", "quit"):
                print("Myllers: Até a próxima!")
                break

            if command in ("historico", "histórico", "memoria", "memória"):
                show_history()
                continue

            answer = knowledge.answer(question)

            if answer is None:
                answer = (
                    "Ainda não tenho conhecimento suficiente para "
                    "responder isso com segurança. Meu modelo neural "
                    "continua em treinamento."
                )
                source = "conhecimento insuficiente"
            else:
                source = "conhecimento"

            print(f"Myllers [{source}]: {answer}")

            memory.remember(question, answer)

    finally:
        memory.close()


if __name__ == "__main__":
    main()
