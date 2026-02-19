import { apiClient } from './client';

export interface SystemConfig {
  id?: number;
  configKey: string;
  configValue: string;
  description?: string;
  category?: string;
  updatedBy?: string;
  createdAt?: string;
  updatedAt?: string;
}

export interface SystemConfigCategory {
  [category: string]: SystemConfig[];
}

export const systemApi = {
  // Get all configs grouped by category
  getAllConfigs: () =>
    apiClient.get<SystemConfigCategory>('/api/v1/admin/system-config'),

  // Get configs by category
  getConfigsByCategory: (category: string) =>
    apiClient.get<SystemConfig[]>(`/api/v1/admin/system-config/category/${category}`),

  // Get single config
  getConfig: (key: string) =>
    apiClient.get<SystemConfig>(`/api/v1/admin/system-config/${key}`),

  // Update config
  updateConfig: (key: string, value: string) =>
    apiClient.put<SystemConfig>(`/api/v1/admin/system-config/${key}`, { value }),

  // Delete config
  deleteConfig: (key: string) =>
    apiClient.delete<void>(`/api/v1/admin/system-config/${key}`),

  // Reset to defaults
  resetToDefaults: () =>
    apiClient.post<void>('/api/v1/admin/system-config/reset'),

  // Initialize defaults
  initializeDefaults: () =>
    apiClient.post<void>('/api/v1/admin/system-config/initialize'),
};
