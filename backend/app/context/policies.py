from __future__ import annotations

from app.context.agent_views import (
    AgentContextView,
    ContextTarget,
    RuntimeSessionContext,
    ScenarioSetupContext,
)
from app.context.schemas import (
    AgentContext,
    SessionContext,
)


class AgentContextPolicyService:
    """
    Apply deterministic agent-specific context visibility rules.

    The policy service never loads or persists state.

    It receives an already validated broad AgentContext and returns
    a filtered request-scoped projection for one runtime consumer.
    """

    POLICY_VERSION = "agent-context-policy-v1"

    def select(
        self,
        *,
        target: ContextTarget,
        context: AgentContext,
    ) -> AgentContextView:
        if target is ContextTarget.ORCHESTRATOR:
            return self._for_orchestrator(context)

        if target is ContextTarget.SCENARIO:
            return self._for_scenario(context)

        if target is ContextTarget.PROCESS_COACH:
            return self._for_process_coach(context)

        if target is ContextTarget.AP_PLUS_NAVIGATOR:
            return self._for_ap_plus_navigator(context)

        raise ValueError(f"Unsupported context target: {target}")

    def _for_orchestrator(
        self,
        context: AgentContext,
    ) -> AgentContextView:
        """
        Give routing only the information needed to understand the
        immediate request and current runtime position.

        Deliberately excluded:
        - full ScenarioCard
        - ScenarioDraft
        - profile
        - progress
        - conversation history
        - adaptations
        - knowledge chunks
        """

        runtime_session = self._build_runtime_session(
            context.session_context,
            include_unresolved_questions=False,
        )

        sections = [
            "current_message",
        ]

        if runtime_session is not None:
            sections.append("runtime_session")

        return AgentContextView(
            target=(ContextTarget.ORCHESTRATOR),
            user_id=context.user_id,
            session_id=context.session_id,
            current_message=(context.current_message),
            runtime_session=runtime_session,
            policy_version=(self.POLICY_VERSION),
            included_sections=sections,
        )

    def _for_scenario(
        self,
        context: AgentContext,
    ) -> AgentContextView:
        """
        Scenario Agent receives confirmed scenario information plus
        temporary onboarding state.

        It does not receive learning progress or normal process-state
        information because those are unrelated to scenario
        collection.
        """

        scenario_setup = self._build_scenario_setup(context.session_context)

        sections = [
            "current_message",
            "guidance_level",
        ]

        if context.scenario_card is not None:
            sections.append("scenario_card")

        if scenario_setup is not None:
            sections.append("scenario_setup")

        if context.recent_messages:
            sections.append("recent_messages")

        return AgentContextView(
            target=ContextTarget.SCENARIO,
            user_id=context.user_id,
            session_id=context.session_id,
            current_message=(context.current_message),
            scenario_card=(context.scenario_card),
            scenario_setup=scenario_setup,
            recent_messages=list(context.recent_messages),
            guidance_level=(context.guidance_level),
            policy_version=(self.POLICY_VERSION),
            included_sections=sections,
        )

    def _for_process_coach(
        self,
        context: AgentContext,
    ) -> AgentContextView:
        """
        Process Coach receives business scenario, learning state,
        progress, runtime process state, guidance, relevant history,
        and future process-knowledge chunks.

        ScenarioDraft is deliberately excluded.
        """

        runtime_session = self._build_runtime_session(
            context.session_context,
            include_unresolved_questions=True,
        )

        sections = [
            "current_message",
            "learning_profile",
            "progress",
            "guidance_level",
        ]

        if context.scenario_card is not None:
            sections.append("scenario_card")

        if runtime_session is not None:
            sections.append("runtime_session")

        if context.recent_messages:
            sections.append("recent_messages")

        if context.session_summary is not None:
            sections.append("session_summary")

        if context.user_adaptations:
            sections.append("user_adaptations")

        if context.knowledge_chunks:
            sections.append("knowledge_chunks")

        return AgentContextView(
            target=(ContextTarget.PROCESS_COACH),
            user_id=context.user_id,
            session_id=context.session_id,
            current_message=(context.current_message),
            scenario_card=(context.scenario_card),
            runtime_session=runtime_session,
            learning_profile=(context.learning_profile),
            progress=context.progress,
            recent_messages=list(context.recent_messages),
            session_summary=(context.session_summary),
            guidance_level=(context.guidance_level),
            user_adaptations=list(context.user_adaptations),
            knowledge_chunks=list(context.knowledge_chunks),
            policy_version=(self.POLICY_VERSION),
            included_sections=sections,
        )

    def _for_ap_plus_navigator(
        self,
        context: AgentContext,
    ) -> AgentContextView:
        """
        AP+ Navigator receives confirmed scenario facts and the
        immediate AP+ runtime position.

        It does not receive ScenarioDraft, learning profile,
        durable progress, or user adaptations.

        AP+-specific retrieved knowledge will later enter through
        knowledge_chunks.
        """

        runtime_session = self._build_runtime_session(
            context.session_context,
            include_unresolved_questions=True,
        )

        sections = [
            "current_message",
            "guidance_level",
        ]

        if context.scenario_card is not None:
            sections.append("scenario_card")

        if runtime_session is not None:
            sections.append("runtime_session")

        if context.recent_messages:
            sections.append("recent_messages")

        if context.session_summary is not None:
            sections.append("session_summary")

        if context.knowledge_chunks:
            sections.append("knowledge_chunks")

        return AgentContextView(
            target=(ContextTarget.AP_PLUS_NAVIGATOR),
            user_id=context.user_id,
            session_id=context.session_id,
            current_message=(context.current_message),
            scenario_card=(context.scenario_card),
            runtime_session=runtime_session,
            recent_messages=list(context.recent_messages),
            session_summary=(context.session_summary),
            guidance_level=(context.guidance_level),
            knowledge_chunks=list(context.knowledge_chunks),
            policy_version=(self.POLICY_VERSION),
            included_sections=sections,
        )

    @staticmethod
    def _build_runtime_session(
        session_context: SessionContext | None,
        *,
        include_unresolved_questions: bool,
    ) -> RuntimeSessionContext | None:
        if session_context is None:
            return None

        unresolved_questions: list[str]

        if include_unresolved_questions:
            unresolved_questions = list(session_context.unresolved_questions)
        else:
            unresolved_questions = []

        return RuntimeSessionContext(
            current_objective=(session_context.current_objective),
            current_step_id=(session_context.current_step_id),
            current_process_phase=(session_context.current_process_phase),
            active_blocker=(session_context.active_blocker),
            last_selected_agent=(session_context.last_selected_agent),
            unresolved_questions=(unresolved_questions),
        )

    @staticmethod
    def _build_scenario_setup(
        session_context: SessionContext | None,
    ) -> ScenarioSetupContext | None:
        if session_context is None:
            return None

        return ScenarioSetupContext(
            status=(session_context.scenario_setup_status),
            draft=(session_context.scenario_draft),
        )
