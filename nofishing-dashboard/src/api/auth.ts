import apiClient from './client';
import type {
  LoginRequest,
  LoginResponse,
  User,
  PaginatedResponse,
} from '../types';

export const authApi = {
  login: (data: LoginRequest) =>
    apiClient.post<LoginResponse>('/api/v1/auth/login', data),

  refreshToken: (refreshToken: string) =>
    apiClient.post<LoginResponse>('/api/v1/auth/refresh', { refreshToken }),

  logout: () =>
    apiClient.post<void>('/api/v1/auth/logout'),

  getCurrentUser: () =>
    apiClient.get<User>('/api/v1/auth/me'),
};

export const userApi = {
  getUsers: (params: { keyword?: string; page?: number; size?: number }) =>
    apiClient.get<PaginatedResponse<User>>('/api/v1/admin/users', { params }),

  getUser: (id: number) =>
    apiClient.get<User>(`/api/v1/admin/users/${id}`),

  createUser: (data: Partial<User> & { password: string }) =>
    apiClient.post<User>('/api/v1/admin/users', data),

  updateUser: (id: number, data: Partial<User>) =>
    apiClient.put<User>(`/api/v1/admin/users/${id}`, data),

  deleteUser: (id: number) =>
    apiClient.delete<void>(`/api/v1/admin/users/${id}`),

  resetPassword: (id: number, newPassword: string) =>
    apiClient.post<void>(`/api/v1/admin/users/${id}/reset-password`, null, {
      params: { newPassword }
    }),

  changePassword: (userId: number, oldPassword: string, newPassword: string) =>
    apiClient.post<void>('/api/v1/auth/change-password', {
      userId, oldPassword, newPassword
    }),
};
