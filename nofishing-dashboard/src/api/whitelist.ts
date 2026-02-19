import apiClient from './client';
import type {
  WhitelistEntry,
  PaginatedResponse,
} from '../types';

export const whitelistApi = {
  getWhitelist: (params: { page?: number; size?: number }) =>
    apiClient.get<PaginatedResponse<WhitelistEntry>>('/api/v1/whitelist', { params }),

  getWhitelistEntry: (id: number) =>
    apiClient.get<WhitelistEntry>(`/api/v1/whitelist/${id}`),

  createWhitelistEntry: (data: Omit<WhitelistEntry, 'id' | 'createdAt'>) =>
    apiClient.post<WhitelistEntry>('/api/v1/whitelist', data),

  updateWhitelistEntry: (id: number, data: Partial<WhitelistEntry>) =>
    apiClient.put<WhitelistEntry>(`/api/v1/whitelist/${id}`, data),

  deleteWhitelistEntry: (id: number) =>
    apiClient.delete<void>(`/api/v1/whitelist/${id}`),

  checkWhitelist: (url: string) =>
    apiClient.get<{ whitelisted: boolean }>('/api/v1/whitelist/check', { params: { url } }),

  existsWhitelist: (pattern: string) =>
    apiClient.get<{ exists: boolean }>('/api/v1/whitelist/exists', { params: { pattern } }),

  batchImport: (data: { patterns: string[]; comment?: string }) =>
    apiClient.post<{ success: boolean; results: string[]; total: number }>('/api/v1/whitelist/batch-import', data),

  batchDelete: (ids: number[]) =>
    apiClient.delete<{ success: boolean; deleted: number }>('/api/v1/whitelist/batch', { data: ids }),
};
