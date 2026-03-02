// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Fraud Detector', 
  tagline: 'Introduction to Fraud Detection', 
  favicon: 'img/favicon.ico',

  future: { v4: true },

  // --- GITHUB PAGES SETTINGS ---
  url: 'https://JackSteve-code.github.io', // Replace with your username
  baseUrl: '/fraud-detector/', // The name of your GitHub repository
  organizationName: 'JackSteve-code', // Replace with your username
  projectName: 'fraud-detector', // The name of your GitHub repository
  deploymentBranch: 'gh-pages',
  trailingSlash: false,
  // -----------------------------

  onBrokenLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          path: 'My-docs', 
          sidebarPath: './sidebars.js',
          routeBasePath: '/', 
          exclude: ['node_modules/**'],
        },
        blog: false, 
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: 'Fraud Detector',
        logo: {
          alt: 'Logo',
          src: 'img/logo.svg',
        },
        items: [],
      },
      footer: {
        style: 'dark',
        links: [], 
        copyright: `Copyright © ${new Date().getFullYear()} Fraud Detector Project.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;