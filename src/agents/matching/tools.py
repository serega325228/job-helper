import uuid
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph


class SupervisorToolHandlers:
    def __init__(self, resume_graph: CompiledStateGraph):
        self._resume_graph = resume_graph

    async def generate_resume(self, profile_id: uuid.UUID, vacancy_id: uuid.UUID) -> dict:
        result = await self._resume_graph.ainvoke({
            "profile_id": profile_id,
            "vacancy_id": vacancy_id,
            "output_format": "both",
        })
        return {
            "status": result["status"],
            "document_id": result.get("document_id"),
            "output_path": result.get("output_path"),
            "warnings": result.get("warnings", []),
        }

def create_supervisor_tools(handlers: SupervisorToolHandlers):
    @tool
    async def generate_resume(profile_id: uuid.UUID, vacancy_id: uuid.UUID):
        """Создать адаптированное резюме для выбранной вакансии"""
        return await handlers.generate_resume(profile_id, vacancy_id)
    return [generate_resume]
