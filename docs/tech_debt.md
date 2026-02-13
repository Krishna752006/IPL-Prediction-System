# Technical Debt

- Training pipeline tightly coupled with API
- No standalone training script
- No dataset versioning
- Model overwritten without evaluation
- Preprocessing objects not saved
- No tests
- No logging
- No CI/CD
- Neural architecture selected before baseline validation (now corrected)
- API handling inference + training violates separation of concerns.
