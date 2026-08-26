interface ApiErrorResponse {
  response?: {
    data?: {
      detail?: string;
      message?: string;
    };
  };
}

const isApiErrorResponse = (error: unknown): error is ApiErrorResponse =>
  typeof error === 'object' && error !== null && 'response' in error;

/**
 * Safely extracts a human-readable message from an unknown error thrown by
 * an axios request, falling back to `fallback` if no message is found.
 */
export const getApiErrorMessage = (
  error: unknown,
  fallback = 'Something went wrong.'
): string => {
  if (isApiErrorResponse(error)) {
    const data = error.response?.data;
    if (data?.detail) return data.detail;
    if (data?.message) return data.message;
  }
  return fallback;
};