import React, { useState } from 'react';
import { Layout, Menu, Dropdown, Avatar, Button, Typography } from 'antd';
import {
  HomeOutlined,
  SafetyOutlined,
  HistoryOutlined,
  BarChartOutlined,
  MonitorOutlined,
  CheckCircleOutlined as ShieldCheckOutlined,
  SecurityScanOutlined as ShieldOutlined,
  UserOutlined,
  LogoutOutlined,
  TeamOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  SettingOutlined,
  FileTextOutlined,
  ApiOutlined,
  SecurityScanOutlined as SecurityIcon,
} from '@ant-design/icons';
import { useNavigate, useLocation, Outlet } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';
import './Layout.css';

const { Header, Sider, Content } = Layout;
const { Text } = Typography;

const AppLayout: React.FC = () => {
  const { user, logout, isAdmin } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [collapsed, setCollapsed] = useState(false);

  const menuItems = [
    { key: '/', icon: <HomeOutlined />, label: '仪表盘' },
    { key: '/detection', icon: <SafetyOutlined />, label: 'URL检测' },
    { key: '/history', icon: <HistoryOutlined />, label: '历史记录' },
    { key: '/statistics', icon: <BarChartOutlined />, label: '统计分析' },
    { key: '/monitoring', icon: <MonitorOutlined />, label: '系统监控' },
    { key: '/whitelist', icon: <ShieldCheckOutlined />, label: '白名单' },
    { key: '/blacklist', icon: <ShieldOutlined />, label: '黑名单' },
  ];

  if (isAdmin()) {
    menuItems.push({ key: '/users', icon: <TeamOutlined />, label: '用户管理' });
    menuItems.push({ key: '/system-config', icon: <SettingOutlined />, label: '系统配置' });
    menuItems.push({ key: '/audit-logs', icon: <FileTextOutlined />, label: '审计日志' });
    menuItems.push({ key: '/api-keys', icon: <ApiOutlined />, label: 'API密钥' });
  }

  const handleClick = ({ key }: { key: string }) => {
    navigate(key);
  };

  const handleLogout = async () => {
    await logout();
  };

  const userMenuItems = [
    {
      key: 'profile',
      icon: <UserOutlined />,
      label: '个人资料',
    },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: '退出登录',
      onClick: handleLogout,
    },
  ];

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider
        collapsible
        collapsed={collapsed}
        onCollapse={setCollapsed}
        theme="light"
      >
        <Menu
          theme="light"
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={handleClick}
          style={{ marginTop: 16 }}
        />
      </Sider>
      <Layout>
        <Header className="header">
          <div className="header-left">
            <SecurityIcon style={{ fontSize: 24, color: '#1890ff', marginRight: 8 }} />
            <Text strong style={{ fontSize: 18 }}>NoFishing</Text>
          </div>
          <Button
            type="text"
            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => setCollapsed(!collapsed)}
            className="collapse-btn-header"
          />
          <div className="header-right">
            <span className="username">{user?.username}</span>
            <Dropdown menu={{ items: userMenuItems }} placement="bottomRight">
              <Avatar icon={<UserOutlined />} style={{ cursor: 'pointer' }} />
            </Dropdown>
          </div>
        </Header>
        <Content className="content">
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  );
};

export default AppLayout;
