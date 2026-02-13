# Technical Debt

This document tracks known architectural gaps, engineering risks, and missing production capabilities within the system.

---

## 🔴 Critical Architecture Risks

These issues directly impact scalability, reliability, and production readiness.

- Training pipeline is tightly coupled with the API.
- API currently handles both inference and model retraining, violating separation of concerns.
- Model is overwritten after retraining with no versioning or rollback capability.
- No standalone training pipeline exists.
- Dataset is not versioned.
- Preprocessing artifacts (encoders, scalers) are not persisted.

---

## 🟠 Backend Infrastructure Gaps

These prevent the system from functioning as a true production service.

- No database connected; application operates fully stateless.
- Authentication is not implemented — login/register are UI-only.
- Predictions are not associated with users.
- No logging mechanism is in place.
- No CI/CD pipeline configured.

---

## 🟡 API Reliability Issues

These affect API correctness and developer experience.

- API does not consistently return HTTP status codes.
- Error handling relies on null responses instead of structured error objects.
- No request validation layer implemented.
- No rate limiting or abuse protection.

---

## 🟢 Engineering Quality Improvements

Important for long-term maintainability and team scalability.

- No automated tests present.
- Neural architecture was selected before establishing baseline models (now corrected).

---

## Priority Guidance

**Fix First (High Impact):**
1. Decouple training from the API.
2. Introduce model versioning.
3. Add database integration.
4. Implement proper authentication.

**Fix Next:**
1. Add logging.
2. Implement structured error handling.
3. Add request validation.
4. Establish CI/CD.

**Fix Later:**
1. Expand automated testing coverage.
2. Add rate limiting.