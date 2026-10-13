from uuid import UUID

from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph


class WizardToolHandlers:
    def __init__(self, resume_graph: CompiledStateGraph):
        self._resume_graph = resume_graph

    async def create_master_resume(self, profile_id: UUID) -> dict:
        result = await self._resume_graph.ainvoke(
            {
                "profile_id": profile_id,
                "output_format": "both",
            }
        )
        return {
            "status": result["status"],
            "document_id": result.get("document_id"),
            "output_path": result.get("output_path"),
            "warnings": result.get("warnings", []),
        }


    async def generate_resume(
        self, profile_id: UUID, vacancy_id: UUID
    ) -> dict:
        result = await self._resume_graph.ainvoke(
            {
                "profile_id": profile_id,
                "vacancy_id": vacancy_id,
                "output_format": "both",
            }
        )
        return {
            "status": result["status"],
            "document_id": result.get("document_id"),
            "output_path": result.get("output_path"),
            "warnings": result.get("warnings", []),
        }




def create_wizard_tools(handlers: WizardToolHandlers):
    @tool
    async def generate_resume(profile_id: UUID, vacancy_id: UUID):
        """Generate a tailored resume for the selected vacancy"""
        return await handlers.generate_resume(profile_id, vacancy_id)

    @tool
    async def create_master_resume(profile_id: UUID):
        """Create a master resume for the profile"""
        return await handlers.create_master_resume(profile_id)

    return [generate_resume, create_master_resume]
