import apiClient from './client';
import type {
  DetectionRequest,
  DetectionResponse,
  PaginatedResponse,
  DetectionHistory,
} from '../types';

export const detectionApi = {
  detect: (data: DetectionRequest) =>
    apiClient.post<DetectionResponse>('/api/v1/detect', data),

  batchDetect: (urls: string[]) =>
    apiClient.post<DetectionResponse[]>('/api/v1/detect/batch', { urls }),
};

export const historyApi = {
  getHistory: (params: {
    keyword?: string;
    isPhishing?: boolean;
    startTime?: string;
    endTime?: string;
    page?: number;
    size?: number;
  }) => apiClient.get<PaginatedResponse<DetectionHistory>>('/api/v1/history', { params }),

  getHistoryById: (id: number) =>
    apiClient.get<DetectionHistory>(`/api/v1/history/${id}`),

  deleteHistory: (id: number) =>
    apiClient.delete<void>(`/api/v1/history/${id}`),

  exportHistory: (params: {
    keyword?: string;
    isPhishing?: boolean;
    startTime?: string;
    endTime?: string;
  }) => {
    return apiClient.get('/api/v1/history/export', {
      params,
      responseType: 'blob',
    }).then((response) => {
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const a = document.createElement('a');
      a.href = url;
      a.download = `检测历史_${Date.now()}.xlsx`;
      a.click();
      window.URL.revokeObjectURL(url);
    });
  },
};
