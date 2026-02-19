import { apiClient } from './client';

export interface ApiKey {
  id: number;
  keyValue: string;
  name: string;
  userId: number;
  username?: string;
  permissions: string[];
  expiresAt?: string;
  lastUsedAt?: string;
  isEnabled: boolean;
  createdBy?: string;
  createdAt?: string;
}

export interface CreateApiKeyRequest {
  name: string;
  userId: number;
  permissions: string[];
  expiresAt?: string;
}

export const apiKeyApi = {
  // Create API key
  createKey: (data: CreateApiKeyRequest) =>
    apiClient.post<ApiKey>('/api/v1/admin/api-keys', data),

  // Get all keys
  getAllKeys: () =>
    apiClient.get<ApiKey[]>('/api/v1/admin/api-keys'),

  // Get keys by user
  getUserKeys: (userId: number) =>
    apiClient.get<ApiKey[]>(`/api/v1/admin/api-keys/user/${userId}`),

  // Get key by ID
  getKeyById: (id: number) =>
    apiClient.get<ApiKey>(`/api/v1/admin/api-keys/${id}`),

  // Revoke key
  revokeKey: (id: number) =>
    apiClient.delete<void>(`/api/v1/admin/api-keys/${id}`),

  // Disable key
  disableKey: (id: number) =>
    apiClient.post<void>(`/api/v1/admin/api-keys/${id}/disable`),

  // Cleanup expired keys
  cleanupExpiredKeys: () =>
    apiClient.post<void>('/api/v1/admin/api-keys/cleanup'),
};
