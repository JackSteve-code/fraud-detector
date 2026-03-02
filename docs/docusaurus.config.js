// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Fraud Detector', 
  tagline: 'The Secret Sauce', 
  favicon: 'img/favicon.ico',
  future: { v4: true },

  url: 'https://JackSteve-code.github.io', 
  baseUrl: '/fraud-detector/', 
  organizationName: 'JackSteve-code', 
  projectName: 'fraud-detector', 
  deploymentBranch: 'gh-pages',
  trailingSlash: false,
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
          sidebarItemsGenerator: async () => [], // FORCE EMPTY SIDEBAR
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
        title: '', // HIDES TEXT
        logo: {
          src: 'img/logo.svg', // Placeholder
          width: 0,            // HIDES LOGO
          height: 0,
        },
        items: [],             // HIDES LINKS/INTRODUCTIONS
      },
      footer: {
        style: 'dark',
        links: [], 
        copyright: ' ',        // REMOVES COPYRIGHT TEXT
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
      colorMode: {
        defaultMode: 'dark',
        disableSwitch: true,   // LOCKS COSMIC PURPLE
        respectPrefersColorScheme: false,
      },
    }),
};

export default config;