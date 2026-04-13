"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import Box from "@mui/material/Box";
import Drawer from "@mui/material/Drawer";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import Typography from "@mui/material/Typography";
import Divider from "@mui/material/Divider";
import DashboardIcon from "@mui/icons-material/Dashboard";
import BarChartIcon from "@mui/icons-material/BarChart";
import SettingsIcon from "@mui/icons-material/Settings";
import ChatIcon from "@mui/icons-material/Chat";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import SmartToyIcon from "@mui/icons-material/SmartToy";

interface SidebarProps {
  width: number;
}

interface SameZoneNavItem {
  label: string;
  href: string;
  icon: React.ReactNode;
  crossZone?: false;
}

interface CrossZoneNavItem {
  label: string;
  href: string;
  icon: React.ReactNode;
  crossZone: true;
}

type NavItem = SameZoneNavItem | CrossZoneNavItem;

const NAV_ITEMS: NavItem[] = [
  { label: "Dashboard", href: "/", icon: <DashboardIcon /> },
  { label: "Monitoring", href: "/monitoring", icon: <BarChartIcon /> },
  { label: "Settings", href: "/settings", icon: <SettingsIcon /> },
  { label: "Chat", href: "/chat", icon: <ChatIcon />, crossZone: true },
  { label: "Upload", href: "/upload", icon: <UploadFileIcon />, crossZone: true },
];

export default function Sidebar({ width }: SidebarProps) {
  const pathname = usePathname();

  return (
    <Drawer
      variant="permanent"
      sx={{
        width,
        flexShrink: 0,
        "& .MuiDrawer-paper": {
          width,
          boxSizing: "border-box",
          borderRight: "1px solid",
          borderColor: "divider",
        },
      }}
    >
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          gap: 1,
          px: 2,
          py: 2.5,
        }}
      >
        <SmartToyIcon color="primary" fontSize="medium" />
        <Typography variant="subtitle1" fontWeight={700} noWrap>
          Bedrock Agents
        </Typography>
      </Box>

      <Divider />

      <List sx={{ pt: 1 }}>
        {NAV_ITEMS.map((item) => {
          const selected = !item.crossZone && pathname === item.href;

          const buttonContent = (
            <ListItemButton
              selected={selected}
              sx={{
                borderRadius: 1,
                mx: 1,
                mb: 0.5,
                "&.Mui-selected": {
                  bgcolor: "primary.main",
                  color: "primary.contrastText",
                  "& .MuiListItemIcon-root": {
                    color: "primary.contrastText",
                  },
                  "&:hover": {
                    bgcolor: "primary.dark",
                  },
                },
              }}
            >
              <ListItemIcon sx={{ minWidth: 36 }}>{item.icon}</ListItemIcon>
              <ListItemText
                primary={item.label}
                primaryTypographyProps={{ fontSize: 14, fontWeight: selected ? 600 : 400 }}
              />
            </ListItemButton>
          );

          return (
            <ListItem key={item.href} disablePadding>
              {item.crossZone ? (
                <Box
                  component="a"
                  href={item.href}
                  sx={{ width: "100%", textDecoration: "none", color: "inherit" }}
                >
                  {buttonContent}
                </Box>
              ) : (
                <Box
                  component={Link}
                  href={item.href}
                  sx={{ width: "100%", textDecoration: "none", color: "inherit" }}
                >
                  {buttonContent}
                </Box>
              )}
            </ListItem>
          );
        })}
      </List>
    </Drawer>
  );
}
