class AgentGuardError(Exception):
    pass


class GuardBlockedError(AgentGuardError):
    def __init__(self, message: str, reason_code: str, shield_name: str) -> None:
        self.reason_code = reason_code
        self.shield_name = shield_name
        super().__init__(f"[{shield_name}] {message} (code: {reason_code})")


class GuardShieldError(AgentGuardError):
    def __init__(self, shield_name: str, detail: str) -> None:
        self.shield_name = shield_name
        super().__init__(f"Shield '{shield_name}' raised an unexpected error: {detail}")


class HumanGateSyncError(AgentGuardError):
    pass


class HumanGateTimeoutError(AgentGuardError):
    pass
