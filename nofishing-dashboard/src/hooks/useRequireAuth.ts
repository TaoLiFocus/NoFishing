import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from './useAuth';

export const useRequireAuth = (requireAdmin = false) => {
  const { isAuthenticated, isAdmin } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (!isAuthenticated) {
      navigate('/login', { replace: true });
    } else if (requireAdmin && !isAdmin()) {
      navigate('/unauthorized', { replace: true });
    }
  }, [isAuthenticated, isAdmin, navigate, requireAdmin]);
};
