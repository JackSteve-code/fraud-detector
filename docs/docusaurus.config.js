// @ts-check
import { themes as prismThemes } from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Fraud Detector',
  tagline: 'The Secret Sauce',
  favicon: 'img/favicon.ico',

  url: 'https://JackSteve-code.github.io',
  baseUrl: '/fraud-detector/',
  organizationName: 'JackSteve-code',
  projectName: 'fraud-detector',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,
  onBrokenLinks: 'warn',

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          path: '.', 
          routeBasePath: '/', 
          sidebarPath: false, 
          exclude: [
            '**/node_modules/**',
            '**/build/**',
            '**/.docusaurus/**',
            'README.md',
          ],
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
        title: '',
        logo: {
          alt: 'Logo',
          src: 'img/logo.svg',
          width: 0,
          height: 0,
        },
        items: [],
      },
      footer: {
        style: 'dark',
        links: [],
        copyright: ' ',
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
      colorMode: {
        defaultMode: 'dark',
        disableSwitch: true,
        respectPrefersColorScheme: false,
      },
    }),

  plugins: [
    () => ({
      name: 'webpack-config-cleanup',
      configureWebpack() {
        return {
          ignoreWarnings: [
            {
              module: /@docusaurus\/mdx-loader/,
              message: /No serializer registered for VFileMessage/,
            },
          ],
        };
      },
    }),
  ],
};

export default config;