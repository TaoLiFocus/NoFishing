import { useAppDispatch, useAppSelector } from '../store/hooks';
import { login as loginAction, logout as logoutAction, fetchCurrentUser } from '../store/slices/authSlice';
import { useNavigate } from 'react-router-dom';
import { message } from 'antd';
import type { LoginRequest } from '../types';

export const useAuth = () => {
  const dispatch = useAppDispatch();
  const navigate = useNavigate();
  const { user, isAuthenticated, isLoading, error } = useAppSelector((state) => state.auth);

  const login = async (credentials: LoginRequest) => {
    try {
      await dispatch(loginAction(credentials)).unwrap();
      message.success('Login successful');
      navigate('/');
    } catch (error) {
      message.error('Login failed: ' + (error as Error).message);
      throw error;
    }
  };

  const logout = async () => {
    try {
      await dispatch(logoutAction()).unwrap();
      message.success('Logged out successfully');
      navigate('/login');
    } catch (error) {
      message.error('Logout failed');
    }
  };

  const refetchUser = async () => {
    try {
      await dispatch(fetchCurrentUser()).unwrap();
    } catch (error) {
      console.error('Failed to fetch user:', error);
    }
  };

  const hasRole = (roles: string[]) => {
    return user?.role ? roles.includes(user.role) : false;
  };

  const isAdmin = () => {
    return user?.role === 'ADMIN';
  };

  return {
    user,
    isAuthenticated,
    isLoading,
    error,
    login,
    logout,
    refetchUser,
    hasRole,
    isAdmin,
  };
};
