import apiClient from './client';
import type {
  BlacklistEntry,
  PaginatedResponse,
} from '../types';

export const blacklistApi = {
  getBlacklist: (params: { page?: number; size?: number }) =>
    apiClient.get<PaginatedResponse<BlacklistEntry>>('/api/v1/blacklist', { params }),

  getBlacklistEntry: (id: number) =>
    apiClient.get<BlacklistEntry>(`/api/v1/blacklist/${id}`),

  createBlacklistEntry: (data: Omit<BlacklistEntry, 'id' | 'createdAt'>) =>
    apiClient.post<BlacklistEntry>('/api/v1/blacklist', data),

  updateBlacklistEntry: (id: number, data: Partial<BlacklistEntry>) =>
    apiClient.put<BlacklistEntry>(`/api/v1/blacklist/${id}`, data),

  deleteBlacklistEntry: (id: number) =>
    apiClient.delete<void>(`/api/v1/blacklist/${id}`),

  checkBlacklist: (url: string) =>
    apiClient.get<{ blacklisted: boolean }>('/api/v1/blacklist/check', { params: { url } }),

  existsBlacklist: (pattern: string) =>
    apiClient.get<{ exists: boolean }>('/api/v1/blacklist/exists', { params: { pattern } }),

  batchImport: (data: { patterns: string[]; comment?: string }) =>
    apiClient.post<{ success: boolean; results: string[]; total: number }>('/api/v1/blacklist/batch-import', data),

  batchDelete: (ids: number[]) =>
    apiClient.delete<{ success: boolean; deleted: number }>('/api/v1/blacklist/batch', { data: ids }),
};
