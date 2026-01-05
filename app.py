<!DOCTYPE html>

<html lang="en">
<head><meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>Untitled35 (1)</title><script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<style type="text/css">
    pre { line-height: 125%; }
td.linenos .normal { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
span.linenos { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
td.linenos .special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
span.linenos.special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
.highlight .hll { background-color: var(--jp-cell-editor-active-background) }
.highlight { background: var(--jp-cell-editor-background); color: var(--jp-mirror-editor-variable-color) }
.highlight .c { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment */
.highlight .err { color: var(--jp-mirror-editor-error-color) } /* Error */
.highlight .k { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword */
.highlight .o { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator */
.highlight .p { color: var(--jp-mirror-editor-punctuation-color) } /* Punctuation */
.highlight .ch { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Multiline */
.highlight .cp { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Preproc */
.highlight .cpf { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Single */
.highlight .cs { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Special */
.highlight .kc { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Pseudo */
.highlight .kr { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Type */
.highlight .m { color: var(--jp-mirror-editor-number-color) } /* Literal.Number */
.highlight .s { color: var(--jp-mirror-editor-string-color) } /* Literal.String */
.highlight .ow { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator.Word */
.highlight .pm { color: var(--jp-mirror-editor-punctuation-color) } /* Punctuation.Marker */
.highlight .w { color: var(--jp-mirror-editor-variable-color) } /* Text.Whitespace */
.highlight .mb { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Bin */
.highlight .mf { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Float */
.highlight .mh { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Hex */
.highlight .mi { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer */
.highlight .mo { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Oct */
.highlight .sa { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Affix */
.highlight .sb { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Backtick */
.highlight .sc { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Char */
.highlight .dl { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Delimiter */
.highlight .sd { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Doc */
.highlight .s2 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Double */
.highlight .se { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Escape */
.highlight .sh { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Heredoc */
.highlight .si { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Interpol */
.highlight .sx { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Other */
.highlight .sr { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Regex */
.highlight .s1 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Single */
.highlight .ss { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Symbol */
.highlight .il { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer.Long */
  </style>
<style type="text/css">
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
 * Mozilla scrollbar styling
 */

/* use standard opaque scrollbars for most nodes */
[data-jp-theme-scrollbars='true'] {
  scrollbar-color: rgb(var(--jp-scrollbar-thumb-color))
    var(--jp-scrollbar-background-color);
}

/* for code nodes, use a transparent style of scrollbar. These selectors
 * will match lower in the tree, and so will override the above */
[data-jp-theme-scrollbars='true'] .CodeMirror-hscrollbar,
[data-jp-theme-scrollbars='true'] .CodeMirror-vscrollbar {
  scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
}

/* tiny scrollbar */

.jp-scrollbar-tiny {
  scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
  scrollbar-width: thin;
}

/* tiny scrollbar */

.jp-scrollbar-tiny::-webkit-scrollbar,
.jp-scrollbar-tiny::-webkit-scrollbar-corner {
  background-color: transparent;
  height: 4px;
  width: 4px;
}

.jp-scrollbar-tiny::-webkit-scrollbar-thumb {
  background: rgba(var(--jp-scrollbar-thumb-color), 0.5);
}

.jp-scrollbar-tiny::-webkit-scrollbar-track:horizontal {
  border-left: 0 solid transparent;
  border-right: 0 solid transparent;
}

.jp-scrollbar-tiny::-webkit-scrollbar-track:vertical {
  border-top: 0 solid transparent;
  border-bottom: 0 solid transparent;
}

/*
 * Lumino
 */

.lm-ScrollBar[data-orientation='horizontal'] {
  min-height: 16px;
  max-height: 16px;
  min-width: 45px;
  border-top: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] {
  min-width: 16px;
  max-width: 16px;
  min-height: 45px;
  border-left: 1px solid #a0a0a0;
}

.lm-ScrollBar-button {
  background-color: #f0f0f0;
  background-position: center center;
  min-height: 15px;
  max-height: 15px;
  min-width: 15px;
  max-width: 15px;
}

.lm-ScrollBar-button:hover {
  background-color: #dadada;
}

.lm-ScrollBar-button.lm-mod-active {
  background-color: #cdcdcd;
}

.lm-ScrollBar-track {
  background: #f0f0f0;
}

.lm-ScrollBar-thumb {
  background: #cdcdcd;
}

.lm-ScrollBar-thumb:hover {
  background: #bababa;
}

.lm-ScrollBar-thumb.lm-mod-active {
  background: #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal'] .lm-ScrollBar-thumb {
  height: 100%;
  min-width: 15px;
  border-left: 1px solid #a0a0a0;
  border-right: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] .lm-ScrollBar-thumb {
  width: 100%;
  min-height: 15px;
  border-top: 1px solid #a0a0a0;
  border-bottom: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-left);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-right);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-up);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-down);
  background-size: 17px;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-Widget {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
}

.lm-Widget.lm-mod-hidden {
  display: none !important;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.lm-AccordionPanel[data-orientation='horizontal'] > .lm-AccordionPanel-title {
  /* Title is rotated for horizontal accordion panel using CSS */
  display: block;
  transform-origin: top left;
  transform: rotate(-90deg) translate(-100%);
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-CommandPalette {
  display: flex;
  flex-direction: column;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-CommandPalette-search {
  flex: 0 0 auto;
}

.lm-CommandPalette-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  min-height: 0;
  overflow: auto;
  list-style-type: none;
}

.lm-CommandPalette-header {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.lm-CommandPalette-item {
  display: flex;
  flex-direction: row;
}

.lm-CommandPalette-itemIcon {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemContent {
  flex: 1 1 auto;
  overflow: hidden;
}

.lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemLabel {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.lm-close-icon {
  border: 1px solid transparent;
  background-color: transparent;
  position: absolute;
  z-index: 1;
  right: 3%;
  top: 0;
  bottom: 0;
  margin: auto;
  padding: 7px 0;
  display: none;
  vertical-align: middle;
  outline: 0;
  cursor: pointer;
}
.lm-close-icon:after {
  content: 'X';
  display: block;
  width: 15px;
  height: 15px;
  text-align: center;
  color: #000;
  font-weight: normal;
  font-size: 12px;
  cursor: pointer;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-DockPanel {
  z-index: 0;
}

.lm-DockPanel-widget {
  z-index: 0;
}

.lm-DockPanel-tabBar {
  z-index: 1;
}

.lm-DockPanel-handle {
  z-index: 2;
}

.lm-DockPanel-handle.lm-mod-hidden {
  display: none !important;
}

.lm-DockPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}

.lm-DockPanel-handle[data-orientation='horizontal'] {
  cursor: ew-resize;
}

.lm-DockPanel-handle[data-orientation='vertical'] {
  cursor: ns-resize;
}

.lm-DockPanel-handle[data-orientation='horizontal']:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}

.lm-DockPanel-handle[data-orientation='vertical']:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}

.lm-DockPanel-overlay {
  z-index: 3;
  box-sizing: border-box;
  pointer-events: none;
}

.lm-DockPanel-overlay.lm-mod-hidden {
  display: none !important;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-Menu {
  z-index: 10000;
  position: absolute;
  white-space: nowrap;
  overflow-x: hidden;
  overflow-y: auto;
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-Menu-content {
  margin: 0;
  padding: 0;
  display: table;
  list-style-type: none;
}

.lm-Menu-item {
  display: table-row;
}

.lm-Menu-item.lm-mod-hidden,
.lm-Menu-item.lm-mod-collapsed {
  display: none !important;
}

.lm-Menu-itemIcon,
.lm-Menu-itemSubmenuIcon {
  display: table-cell;
  text-align: center;
}

.lm-Menu-itemLabel {
  display: table-cell;
  text-align: left;
}

.lm-Menu-itemShortcut {
  display: table-cell;
  text-align: right;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-MenuBar {
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-MenuBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: row;
  list-style-type: none;
}

.lm-MenuBar-item {
  box-sizing: border-box;
}

.lm-MenuBar-itemIcon,
.lm-MenuBar-itemLabel {
  display: inline-block;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-ScrollBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-ScrollBar[data-orientation='horizontal'] {
  flex-direction: row;
}

.lm-ScrollBar[data-orientation='vertical'] {
  flex-direction: column;
}

.lm-ScrollBar-button {
  box-sizing: border-box;
  flex: 0 0 auto;
}

.lm-ScrollBar-track {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
  flex: 1 1 auto;
}

.lm-ScrollBar-thumb {
  box-sizing: border-box;
  position: absolute;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-SplitPanel-child {
  z-index: 0;
}

.lm-SplitPanel-handle {
  z-index: 1;
}

.lm-SplitPanel-handle.lm-mod-hidden {
  display: none !important;
}

.lm-SplitPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}

.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle {
  cursor: ew-resize;
}

.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle {
  cursor: ns-resize;
}

.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}

.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-TabBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-TabBar[data-orientation='horizontal'] {
  flex-direction: row;
  align-items: flex-end;
}

.lm-TabBar[data-orientation='vertical'] {
  flex-direction: column;
  align-items: flex-end;
}

.lm-TabBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex: 1 1 auto;
  list-style-type: none;
}

.lm-TabBar[data-orientation='horizontal'] > .lm-TabBar-content {
  flex-direction: row;
}

.lm-TabBar[data-orientation='vertical'] > .lm-TabBar-content {
  flex-direction: column;
}

.lm-TabBar-tab {
  display: flex;
  flex-direction: row;
  box-sizing: border-box;
  overflow: hidden;
  touch-action: none; /* Disable native Drag/Drop */
}

.lm-TabBar-tabIcon,
.lm-TabBar-tabCloseIcon {
  flex: 0 0 auto;
}

.lm-TabBar-tabLabel {
  flex: 1 1 auto;
  overflow: hidden;
  white-space: nowrap;
}

.lm-TabBar-tabInput {
  user-select: all;
  width: 100%;
  box-sizing: border-box;
}

.lm-TabBar-tab.lm-mod-hidden {
  display: none !important;
}

.lm-TabBar-addButton.lm-mod-hidden {
  display: none !important;
}

.lm-TabBar.lm-mod-dragging .lm-TabBar-tab {
  position: relative;
}

.lm-TabBar.lm-mod-dragging[data-orientation='horizontal'] .lm-TabBar-tab {
  left: 0;
  transition: left 150ms ease;
}

.lm-TabBar.lm-mod-dragging[data-orientation='vertical'] .lm-TabBar-tab {
  top: 0;
  transition: top 150ms ease;
}

.lm-TabBar.lm-mod-dragging .lm-TabBar-tab.lm-mod-dragging {
  transition: none;
}

.lm-TabBar-tabLabel .lm-TabBar-tabInput {
  user-select: all;
  width: 100%;
  box-sizing: border-box;
  background: inherit;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-TabPanel-tabBar {
  z-index: 1;
}

.lm-TabPanel-stackedPanel {
  z-index: 0;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapse {
  display: flex;
  flex-direction: column;
  align-items: stretch;
}

.jp-Collapse-header {
  padding: 1px 12px;
  background-color: var(--jp-layout-color1);
  border-bottom: solid var(--jp-border-width) var(--jp-border-color2);
  color: var(--jp-ui-font-color1);
  cursor: pointer;
  display: flex;
  align-items: center;
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  text-transform: uppercase;
  user-select: none;
}

.jp-Collapser-icon {
  height: 16px;
}

.jp-Collapse-header-collapsed .jp-Collapser-icon {
  transform: rotate(-90deg);
  margin: auto 0;
}

.jp-Collapser-title {
  line-height: 25px;
}

.jp-Collapse-contents {
  padding: 0 12px;
  background-color: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  overflow: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensureUiComponents() in @jupyterlab/buildutils */

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

/* Icons urls */

:root {
  --jp-icon-add-above: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGcgY2xpcC1wYXRoPSJ1cmwoI2NsaXAwXzEzN18xOTQ5MikiPgo8cGF0aCBjbGFzcz0ianAtaWNvbjMiIGQ9Ik00Ljc1IDQuOTMwNjZINi42MjVWNi44MDU2NkM2LjYyNSA3LjAxMTkxIDYuNzkzNzUgNy4xODA2NiA3IDcuMTgwNjZDNy4yMDYyNSA3LjE4MDY2IDcuMzc1IDcuMDExOTEgNy4zNzUgNi44MDU2NlY0LjkzMDY2SDkuMjVDOS40NTYyNSA0LjkzMDY2IDkuNjI1IDQuNzYxOTEgOS42MjUgNC41NTU2NkM5LjYyNSA0LjM0OTQxIDkuNDU2MjUgNC4xODA2NiA5LjI1IDQuMTgwNjZINy4zNzVWMi4zMDU2NkM3LjM3NSAyLjA5OTQxIDcuMjA2MjUgMS45MzA2NiA3IDEuOTMwNjZDNi43OTM3NSAxLjkzMDY2IDYuNjI1IDIuMDk5NDEgNi42MjUgMi4zMDU2NlY0LjE4MDY2SDQuNzVDNC41NDM3NSA0LjE4MDY2IDQuMzc1IDQuMzQ5NDEgNC4zNzUgNC41NTU2NkM0LjM3NSA0Ljc2MTkxIDQuNTQzNzUgNC45MzA2NiA0Ljc1IDQuOTMwNjZaIiBmaWxsPSIjNjE2MTYxIiBzdHJva2U9IiM2MTYxNjEiIHN0cm9rZS13aWR0aD0iMC43Ii8+CjwvZz4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGNsaXAtcnVsZT0iZXZlbm9kZCIgZD0iTTExLjUgOS41VjExLjVMMi41IDExLjVWOS41TDExLjUgOS41Wk0xMiA4QzEyLjU1MjMgOCAxMyA4LjQ0NzcyIDEzIDlWMTJDMTMgMTIuNTUyMyAxMi41NTIzIDEzIDEyIDEzTDIgMTNDMS40NDc3MiAxMyAxIDEyLjU1MjMgMSAxMlY5QzEgOC40NDc3MiAxLjQ0NzcxIDggMiA4TDEyIDhaIiBmaWxsPSIjNjE2MTYxIi8+CjxkZWZzPgo8Y2xpcFBhdGggaWQ9ImNsaXAwXzEzN18xOTQ5MiI+CjxyZWN0IGNsYXNzPSJqcC1pY29uMyIgd2lkdGg9IjYiIGhlaWdodD0iNiIgZmlsbD0id2hpdGUiIHRyYW5zZm9ybT0ibWF0cml4KC0xIDAgMCAxIDEwIDEuNTU1NjYpIi8+CjwvY2xpcFBhdGg+CjwvZGVmcz4KPC9zdmc+Cg==);
  --jp-icon-add-below: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGcgY2xpcC1wYXRoPSJ1cmwoI2NsaXAwXzEzN18xOTQ5OCkiPgo8cGF0aCBjbGFzcz0ianAtaWNvbjMiIGQ9Ik05LjI1IDEwLjA2OTNMNy4zNzUgMTAuMDY5M0w3LjM3NSA4LjE5NDM0QzcuMzc1IDcuOTg4MDkgNy4yMDYyNSA3LjgxOTM0IDcgNy44MTkzNEM2Ljc5Mzc1IDcuODE5MzQgNi42MjUgNy45ODgwOSA2LjYyNSA4LjE5NDM0TDYuNjI1IDEwLjA2OTNMNC43NSAxMC4wNjkzQzQuNTQzNzUgMTAuMDY5MyA0LjM3NSAxMC4yMzgxIDQuMzc1IDEwLjQ0NDNDNC4zNzUgMTAuNjUwNiA0LjU0Mzc1IDEwLjgxOTMgNC43NSAxMC44MTkzTDYuNjI1IDEwLjgxOTNMNi42MjUgMTIuNjk0M0M2LjYyNSAxMi45MDA2IDYuNzkzNzUgMTMuMDY5MyA3IDEzLjA2OTNDNy4yMDYyNSAxMy4wNjkzIDcuMzc1IDEyLjkwMDYgNy4zNzUgMTIuNjk0M0w3LjM3NSAxMC44MTkzTDkuMjUgMTAuODE5M0M5LjQ1NjI1IDEwLjgxOTMgOS42MjUgMTAuNjUwNiA5LjYyNSAxMC40NDQzQzkuNjI1IDEwLjIzODEgOS40NTYyNSAxMC4wNjkzIDkuMjUgMTAuMDY5M1oiIGZpbGw9IiM2MTYxNjEiIHN0cm9rZT0iIzYxNjE2MSIgc3Ryb2tlLXdpZHRoPSIwLjciLz4KPC9nPgo8cGF0aCBjbGFzcz0ianAtaWNvbjMiIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJNMi41IDUuNUwyLjUgMy41TDExLjUgMy41TDExLjUgNS41TDIuNSA1LjVaTTIgN0MxLjQ0NzcyIDcgMSA2LjU1MjI4IDEgNkwxIDNDMSAyLjQ0NzcyIDEuNDQ3NzIgMiAyIDJMMTIgMkMxMi41NTIzIDIgMTMgMi40NDc3MiAxMyAzTDEzIDZDMTMgNi41NTIyOSAxMi41NTIzIDcgMTIgN0wyIDdaIiBmaWxsPSIjNjE2MTYxIi8+CjxkZWZzPgo8Y2xpcFBhdGggaWQ9ImNsaXAwXzEzN18xOTQ5OCI+CjxyZWN0IGNsYXNzPSJqcC1pY29uMyIgd2lkdGg9IjYiIGhlaWdodD0iNiIgZmlsbD0id2hpdGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMS43NDg0NmUtMDcgMS43NDg0NmUtMDcgLTEgNCAxMy40NDQzKSIvPgo8L2NsaXBQYXRoPgo8L2RlZnM+Cjwvc3ZnPgo=);
  --jp-icon-add: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDEzaC02djZoLTJ2LTZINXYtMmg2VjVoMnY2aDZ2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-bell: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE2IDE2IiB2ZXJzaW9uPSIxLjEiPgogICA8cGF0aCBjbGFzcz0ianAtaWNvbjIganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMzMzMzMzIgogICAgICBkPSJtOCAwLjI5Yy0xLjQgMC0yLjcgMC43My0zLjYgMS44LTEuMiAxLjUtMS40IDMuNC0xLjUgNS4yLTAuMTggMi4yLTAuNDQgNC0yLjMgNS4zbDAuMjggMS4zaDVjMC4wMjYgMC42NiAwLjMyIDEuMSAwLjcxIDEuNSAwLjg0IDAuNjEgMiAwLjYxIDIuOCAwIDAuNTItMC40IDAuNi0xIDAuNzEtMS41aDVsMC4yOC0xLjNjLTEuOS0wLjk3LTIuMi0zLjMtMi4zLTUuMy0wLjEzLTEuOC0wLjI2LTMuNy0xLjUtNS4yLTAuODUtMS0yLjItMS44LTMuNi0xLjh6bTAgMS40YzAuODggMCAxLjkgMC41NSAyLjUgMS4zIDAuODggMS4xIDEuMSAyLjcgMS4yIDQuNCAwLjEzIDEuNyAwLjIzIDMuNiAxLjMgNS4yaC0xMGMxLjEtMS42IDEuMi0zLjQgMS4zLTUuMiAwLjEzLTEuNyAwLjMtMy4zIDEuMi00LjQgMC41OS0wLjcyIDEuNi0xLjMgMi41LTEuM3ptLTAuNzQgMTJoMS41Yy0wLjAwMTUgMC4yOCAwLjAxNSAwLjc5LTAuNzQgMC43OS0wLjczIDAuMDAxNi0wLjcyLTAuNTMtMC43NC0wLjc5eiIgLz4KPC9zdmc+Cg==);
  --jp-icon-bug-dot: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiPgogICAgICAgIDxwYXRoIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJNMTcuMTkgOEgyMFYxMEgxNy45MUMxNy45NiAxMC4zMyAxOCAxMC42NiAxOCAxMVYxMkgyMFYxNEgxOC41SDE4VjE0LjAyNzVDMTUuNzUgMTQuMjc2MiAxNCAxNi4xODM3IDE0IDE4LjVDMTQgMTkuMjA4IDE0LjE2MzUgMTkuODc3OSAxNC40NTQ5IDIwLjQ3MzlDMTMuNzA2MyAyMC44MTE3IDEyLjg3NTcgMjEgMTIgMjFDOS43OCAyMSA3Ljg1IDE5Ljc5IDYuODEgMThINFYxNkg2LjA5QzYuMDQgMTUuNjcgNiAxNS4zNCA2IDE1VjE0SDRWMTJINlYxMUM2IDEwLjY2IDYuMDQgMTAuMzMgNi4wOSAxMEg0VjhINi44MUM3LjI2IDcuMjIgNy44OCA2LjU1IDguNjIgNi4wNEw3IDQuNDFMOC40MSAzTDEwLjU5IDUuMTdDMTEuMDQgNS4wNiAxMS41MSA1IDEyIDVDMTIuNDkgNSAxMi45NiA1LjA2IDEzLjQyIDUuMTdMMTUuNTkgM0wxNyA0LjQxTDE1LjM3IDYuMDRDMTYuMTIgNi41NSAxNi43NCA3LjIyIDE3LjE5IDhaTTEwIDE2SDE0VjE0SDEwVjE2Wk0xMCAxMkgxNFYxMEgxMFYxMloiIGZpbGw9IiM2MTYxNjEiLz4KICAgICAgICA8cGF0aCBkPSJNMjIgMTguNUMyMiAyMC40MzMgMjAuNDMzIDIyIDE4LjUgMjJDMTYuNTY3IDIyIDE1IDIwLjQzMyAxNSAxOC41QzE1IDE2LjU2NyAxNi41NjcgMTUgMTguNSAxNUMyMC40MzMgMTUgMjIgMTYuNTY3IDIyIDE4LjVaIiBmaWxsPSIjNjE2MTYxIi8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-bug: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0yMCA4aC0yLjgxYy0uNDUtLjc4LTEuMDctMS40NS0xLjgyLTEuOTZMMTcgNC40MSAxNS41OSAzbC0yLjE3IDIuMTdDMTIuOTYgNS4wNiAxMi40OSA1IDEyIDVjLS40OSAwLS45Ni4wNi0xLjQxLjE3TDguNDEgMyA3IDQuNDFsMS42MiAxLjYzQzcuODggNi41NSA3LjI2IDcuMjIgNi44MSA4SDR2MmgyLjA5Yy0uMDUuMzMtLjA5LjY2LS4wOSAxdjFINHYyaDJ2MWMwIC4zNC4wNC42Ny4wOSAxSDR2MmgyLjgxYzEuMDQgMS43OSAyLjk3IDMgNS4xOSAzczQuMTUtMS4yMSA1LjE5LTNIMjB2LTJoLTIuMDljLjA1LS4zMy4wOS0uNjYuMDktMXYtMWgydi0yaC0ydi0xYzAtLjM0LS4wNC0uNjctLjA5LTFIMjBWOHptLTYgOGgtNHYtMmg0djJ6bTAtNGgtNHYtMmg0djJ6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-build: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE0LjkgMTcuNDVDMTYuMjUgMTcuNDUgMTcuMzUgMTYuMzUgMTcuMzUgMTVDMTcuMzUgMTMuNjUgMTYuMjUgMTIuNTUgMTQuOSAxMi41NUMxMy41NCAxMi41NSAxMi40NSAxMy42NSAxMi40NSAxNUMxMi40NSAxNi4zNSAxMy41NCAxNy40NSAxNC45IDE3LjQ1Wk0yMC4xIDE1LjY4TDIxLjU4IDE2Ljg0QzIxLjcxIDE2Ljk1IDIxLjc1IDE3LjEzIDIxLjY2IDE3LjI5TDIwLjI2IDE5LjcxQzIwLjE3IDE5Ljg2IDIwIDE5LjkyIDE5LjgzIDE5Ljg2TDE4LjA5IDE5LjE2QzE3LjczIDE5LjQ0IDE3LjMzIDE5LjY3IDE2LjkxIDE5Ljg1TDE2LjY0IDIxLjdDMTYuNjIgMjEuODcgMTYuNDcgMjIgMTYuMyAyMkgxMy41QzEzLjMyIDIyIDEzLjE4IDIxLjg3IDEzLjE1IDIxLjdMMTIuODkgMTkuODVDMTIuNDYgMTkuNjcgMTIuMDcgMTkuNDQgMTEuNzEgMTkuMTZMOS45NjAwMiAxOS44NkM5LjgxMDAyIDE5LjkyIDkuNjIwMDIgMTkuODYgOS41NDAwMiAxOS43MUw4LjE0MDAyIDE3LjI5QzguMDUwMDIgMTcuMTMgOC4wOTAwMiAxNi45NSA4LjIyMDAyIDE2Ljg0TDkuNzAwMDIgMTUuNjhMOS42NTAwMSAxNUw5LjcwMDAyIDE0LjMxTDguMjIwMDIgMTMuMTZDOC4wOTAwMiAxMy4wNSA4LjA1MDAyIDEyLjg2IDguMTQwMDIgMTIuNzFMOS41NDAwMiAxMC4yOUM5LjYyMDAyIDEwLjEzIDkuODEwMDIgMTAuMDcgOS45NjAwMiAxMC4xM0wxMS43MSAxMC44NEMxMi4wNyAxMC41NiAxMi40NiAxMC4zMiAxMi44OSAxMC4xNUwxMy4xNSA4LjI4OTk4QzEzLjE4IDguMTI5OTggMTMuMzIgNy45OTk5OCAxMy41IDcuOTk5OThIMTYuM0MxNi40NyA3Ljk5OTk4IDE2LjYyIDguMTI5OTggMTYuNjQgOC4yODk5OEwxNi45MSAxMC4xNUMxNy4zMyAxMC4zMiAxNy43MyAxMC41NiAxOC4wOSAxMC44NEwxOS44MyAxMC4xM0MyMCAxMC4wNyAyMC4xNyAxMC4xMyAyMC4yNiAxMC4yOUwyMS42NiAxMi43MUMyMS43NSAxMi44NiAyMS43MSAxMy4wNSAyMS41OCAxMy4xNkwyMC4xIDE0LjMxTDIwLjE1IDE1TDIwLjEgMTUuNjhaIi8+CiAgICA8cGF0aCBkPSJNNy4zMjk2NiA3LjQ0NDU0QzguMDgzMSA3LjAwOTU0IDguMzM5MzIgNi4wNTMzMiA3LjkwNDMyIDUuMjk5ODhDNy40NjkzMiA0LjU0NjQzIDYuNTA4MSA0LjI4MTU2IDUuNzU0NjYgNC43MTY1NkM1LjM5MTc2IDQuOTI2MDggNS4xMjY5NSA1LjI3MTE4IDUuMDE4NDkgNS42NzU5NEM0LjkxMDA0IDYuMDgwNzEgNC45NjY4MiA2LjUxMTk4IDUuMTc2MzQgNi44NzQ4OEM1LjYxMTM0IDcuNjI4MzIgNi41NzYyMiA3Ljg3OTU0IDcuMzI5NjYgNy40NDQ1NFpNOS42NTcxOCA0Ljc5NTkzTDEwLjg2NzIgNC45NTE3OUMxMC45NjI4IDQuOTc3NDEgMTEuMDQwMiA1LjA3MTMzIDExLjAzODIgNS4xODc5M0wxMS4wMzg4IDYuOTg4OTNDMTEuMDQ1NSA3LjEwMDU0IDEwLjk2MTYgNy4xOTUxOCAxMC44NTUgNy4yMTA1NEw5LjY2MDAxIDcuMzgwODNMOS4yMzkxNSA4LjEzMTg4TDkuNjY5NjEgOS4yNTc0NUM5LjcwNzI5IDkuMzYyNzEgOS42NjkzNCA5LjQ3Njk5IDkuNTc0MDggOS41MzE5OUw4LjAxNTIzIDEwLjQzMkM3LjkxMTMxIDEwLjQ5MiA3Ljc5MzM3IDEwLjQ2NzcgNy43MjEwNSAxMC4zODI0TDYuOTg3NDggOS40MzE4OEw2LjEwOTMxIDkuNDMwODNMNS4zNDcwNCAxMC4zOTA1QzUuMjg5MDkgMTAuNDcwMiA1LjE3MzgzIDEwLjQ5MDUgNS4wNzE4NyAxMC40MzM5TDMuNTEyNDUgOS41MzI5M0MzLjQxMDQ5IDkuNDc2MzMgMy4zNzY0NyA5LjM1NzQxIDMuNDEwNzUgOS4yNTY3OUwzLjg2MzQ3IDguMTQwOTNMMy42MTc0OSA3Ljc3NDg4TDMuNDIzNDcgNy4zNzg4M0wyLjIzMDc1IDcuMjEyOTdDMi4xMjY0NyA3LjE5MjM1IDIuMDQwNDkgNy4xMDM0MiAyLjA0MjQ1IDYuOTg2ODJMMi4wNDE4NyA1LjE4NTgyQzIuMDQzODMgNS4wNjkyMiAyLjExOTA5IDQuOTc5NTggMi4yMTcwNCA0Ljk2OTIyTDMuNDIwNjUgNC43OTM5M0wzLjg2NzQ5IDQuMDI3ODhMMy40MTEwNSAyLjkxNzMxQzMuMzczMzcgMi44MTIwNCAzLjQxMTMxIDIuNjk3NzYgMy41MTUyMyAyLjYzNzc2TDUuMDc0MDggMS43Mzc3NkM1LjE2OTM0IDEuNjgyNzYgNS4yODcyOSAxLjcwNzA0IDUuMzU5NjEgMS43OTIzMUw2LjExOTE1IDIuNzI3ODhMNi45ODAwMSAyLjczODkzTDcuNzI0OTYgMS43ODkyMkM3Ljc5MTU2IDEuNzA0NTggNy45MTU0OCAxLjY3OTIyIDguMDA4NzkgMS43NDA4Mkw5LjU2ODIxIDIuNjQxODJDOS42NzAxNyAyLjY5ODQyIDkuNzEyODUgMi44MTIzNCA5LjY4NzIzIDIuOTA3OTdMOS4yMTcxOCA0LjAzMzgzTDkuNDYzMTYgNC4zOTk4OEw5LjY1NzE4IDQuNzk1OTNaIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iOS45LDEzLjYgMy42LDcuNCA0LjQsNi42IDkuOSwxMi4yIDE1LjQsNi43IDE2LjEsNy40ICIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNS45TDksOS43bDMuOC0zLjhsMS4yLDEuMmwtNC45LDVsLTQuOS01TDUuMiw1Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNy41TDksMTEuMmwzLjgtMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-left: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik0xMC44LDEyLjhMNy4xLDlsMy44LTMuOGwwLDcuNkgxMC44eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-right: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik03LjIsNS4yTDEwLjksOWwtMy44LDMuOFY1LjJINy4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-up-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iMTUuNCwxMy4zIDkuOSw3LjcgNC40LDEzLjIgMy42LDEyLjUgOS45LDYuMyAxNi4xLDEyLjYgIi8+Cgk8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-up: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik01LjIsMTAuNUw5LDYuOGwzLjgsMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-case-sensitive: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgogIDxnIGNsYXNzPSJqcC1pY29uLWFjY2VudDIiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTcuNiw4aDAuOWwzLjUsOGgtMS4xTDEwLDE0SDZsLTAuOSwySDRMNy42LDh6IE04LDkuMUw2LjQsMTNoMy4yTDgsOS4xeiIvPgogICAgPHBhdGggZD0iTTE2LjYsOS44Yy0wLjIsMC4xLTAuNCwwLjEtMC43LDAuMWMtMC4yLDAtMC40LTAuMS0wLjYtMC4yYy0wLjEtMC4xLTAuMi0wLjQtMC4yLTAuNyBjLTAuMywwLjMtMC42LDAuNS0wLjksMC43Yy0wLjMsMC4xLTAuNywwLjItMS4xLDAuMmMtMC4zLDAtMC41LDAtMC43LTAuMWMtMC4yLTAuMS0wLjQtMC4yLTAuNi0wLjNjLTAuMi0wLjEtMC4zLTAuMy0wLjQtMC41IGMtMC4xLTAuMi0wLjEtMC40LTAuMS0wLjdjMC0wLjMsMC4xLTAuNiwwLjItMC44YzAuMS0wLjIsMC4zLTAuNCwwLjQtMC41QzEyLDcsMTIuMiw2LjksMTIuNSw2LjhjMC4yLTAuMSwwLjUtMC4xLDAuNy0wLjIgYzAuMy0wLjEsMC41LTAuMSwwLjctMC4xYzAuMiwwLDAuNC0wLjEsMC42LTAuMWMwLjIsMCwwLjMtMC4xLDAuNC0wLjJjMC4xLTAuMSwwLjItMC4yLDAuMi0wLjRjMC0xLTEuMS0xLTEuMy0xIGMtMC40LDAtMS40LDAtMS40LDEuMmgtMC45YzAtMC40LDAuMS0wLjcsMC4yLTFjMC4xLTAuMiwwLjMtMC40LDAuNS0wLjZjMC4yLTAuMiwwLjUtMC4zLDAuOC0wLjNDMTMuMyw0LDEzLjYsNCwxMy45LDQgYzAuMywwLDAuNSwwLDAuOCwwLjFjMC4zLDAsMC41LDAuMSwwLjcsMC4yYzAuMiwwLjEsMC40LDAuMywwLjUsMC41QzE2LDUsMTYsNS4yLDE2LDUuNnYyLjljMCwwLjIsMCwwLjQsMCwwLjUgYzAsMC4xLDAuMSwwLjIsMC4zLDAuMmMwLjEsMCwwLjIsMCwwLjMsMFY5Ljh6IE0xNS4yLDYuOWMtMS4yLDAuNi0zLjEsMC4yLTMuMSwxLjRjMCwxLjQsMy4xLDEsMy4xLTAuNVY2Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-check: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik05IDE2LjE3TDQuODMgMTJsLTEuNDIgMS40MUw5IDE5IDIxIDdsLTEuNDEtMS40MXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-circle-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyIDJDNi40NyAyIDIgNi40NyAyIDEyczQuNDcgMTAgMTAgMTAgMTAtNC40NyAxMC0xMFMxNy41MyAyIDEyIDJ6bTAgMThjLTQuNDEgMC04LTMuNTktOC04czMuNTktOCA4LTggOCAzLjU5IDggOC0zLjU5IDgtOCA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-circle: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iOSIgY3k9IjkiIHI9IjgiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-clear: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8bWFzayBpZD0iZG9udXRIb2xlIj4KICAgIDxyZWN0IHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgZmlsbD0id2hpdGUiIC8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSI4IiBmaWxsPSJibGFjayIvPgogIDwvbWFzaz4KCiAgPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxyZWN0IGhlaWdodD0iMTgiIHdpZHRoPSIyIiB4PSIxMSIgeT0iMyIgdHJhbnNmb3JtPSJyb3RhdGUoMzE1LCAxMiwgMTIpIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgbWFzaz0idXJsKCNkb251dEhvbGUpIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-close: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1ub25lIGpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIGpwLWljb24zLWhvdmVyIiBmaWxsPSJub25lIj4KICAgIDxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjExIi8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIGpwLWljb24tYWNjZW50Mi1ob3ZlciIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMTkgNi40MUwxNy41OSA1IDEyIDEwLjU5IDYuNDEgNSA1IDYuNDEgMTAuNTkgMTIgNSAxNy41OSA2LjQxIDE5IDEyIDEzLjQxIDE3LjU5IDE5IDE5IDE3LjU5IDEzLjQxIDEyeiIvPgogIDwvZz4KCiAgPGcgY2xhc3M9ImpwLWljb24tbm9uZSBqcC1pY29uLWJ1c3kiIGZpbGw9Im5vbmUiPgogICAgPGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iNyIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-code-check: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBzaGFwZS1yZW5kZXJpbmc9Imdlb21ldHJpY1ByZWNpc2lvbiI+CiAgICA8cGF0aCBkPSJNNi41OSwzLjQxTDIsOEw2LjU5LDEyLjZMOCwxMS4xOEw0LjgyLDhMOCw0LjgyTDYuNTksMy40MU0xMi40MSwzLjQxTDExLDQuODJMMTQuMTgsOEwxMSwxMS4xOEwxMi40MSwxMi42TDE3LDhMMTIuNDEsMy40MU0yMS41OSwxMS41OUwxMy41LDE5LjY4TDkuODMsMTZMOC40MiwxNy40MUwxMy41LDIyLjVMMjMsMTNMMjEuNTksMTEuNTlaIiAvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-code: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIiIGhlaWdodD0iMjIiIHZpZXdCb3g9IjAgMCAyOCAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTExLjQgMTguNkw2LjggMTRMMTEuNCA5LjRMMTAgOEw0IDE0TDEwIDIwTDExLjQgMTguNlpNMTYuNiAxOC42TDIxLjIgMTRMMTYuNiA5LjRMMTggOEwyNCAxNEwxOCAyMEwxNi42IDE4LjZWMTguNloiLz4KCTwvZz4KPC9zdmc+Cg==);
  --jp-icon-collapse-all: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGgKICAgICAgICAgICAgZD0iTTggMmMxIDAgMTEgMCAxMiAwczIgMSAyIDJjMCAxIDAgMTEgMCAxMnMwIDItMiAyQzIwIDE0IDIwIDQgMjAgNFMxMCA0IDYgNGMwLTIgMS0yIDItMnoiIC8+CiAgICAgICAgPHBhdGgKICAgICAgICAgICAgZD0iTTE4IDhjMC0xLTEtMi0yLTJTNSA2IDQgNnMtMiAxLTIgMmMwIDEgMCAxMSAwIDEyczEgMiAyIDJjMSAwIDExIDAgMTIgMHMyLTEgMi0yYzAtMSAwLTExIDAtMTJ6bS0yIDB2MTJINFY4eiIgLz4KICAgICAgICA8cGF0aCBkPSJNNiAxM3YyaDh2LTJ6IiAvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-console: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwMCAyMDAiPgogIDxnIGNsYXNzPSJqcC1jb25zb2xlLWljb24tYmFja2dyb3VuZC1jb2xvciBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMjg4RDEiPgogICAgPHBhdGggZD0iTTIwIDE5LjhoMTYwdjE1OS45SDIweiIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtY29uc29sZS1pY29uLWNvbG9yIGpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIiBmaWxsPSIjZmZmIj4KICAgIDxwYXRoIGQ9Ik0xMDUgMTI3LjNoNDB2MTIuOGgtNDB6TTUxLjEgNzdMNzQgOTkuOWwtMjMuMyAyMy4zIDEwLjUgMTAuNSAyMy4zLTIzLjNMOTUgOTkuOSA4NC41IDg5LjQgNjEuNiA2Ni41eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-copy: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTExLjksMUgzLjJDMi40LDEsMS43LDEuNywxLjcsMi41djEwLjJoMS41VjIuNWg4LjdWMXogTTE0LjEsMy45aC04Yy0wLjgsMC0xLjUsMC43LTEuNSwxLjV2MTAuMmMwLDAuOCwwLjcsMS41LDEuNSwxLjVoOCBjMC44LDAsMS41LTAuNywxLjUtMS41VjUuNEMxNS41LDQuNiwxNC45LDMuOSwxNC4xLDMuOXogTTE0LjEsMTUuNWgtOFY1LjRoOFYxNS41eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-copyright: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGVuYWJsZS1iYWNrZ3JvdW5kPSJuZXcgMCAwIDI0IDI0IiBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCI+CiAgPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0xMS44OCw5LjE0YzEuMjgsMC4wNiwxLjYxLDEuMTUsMS42MywxLjY2aDEuNzljLTAuMDgtMS45OC0xLjQ5LTMuMTktMy40NS0zLjE5QzkuNjQsNy42MSw4LDksOCwxMi4xNCBjMCwxLjk0LDAuOTMsNC4yNCwzLjg0LDQuMjRjMi4yMiwwLDMuNDEtMS42NSwzLjQ0LTIuOTVoLTEuNzljLTAuMDMsMC41OS0wLjQ1LDEuMzgtMS42MywxLjQ0QzEwLjU1LDE0LjgzLDEwLDEzLjgxLDEwLDEyLjE0IEMxMCw5LjI1LDExLjI4LDkuMTYsMTEuODgsOS4xNHogTTEyLDJDNi40OCwyLDIsNi40OCwyLDEyczQuNDgsMTAsMTAsMTBzMTAtNC40OCwxMC0xMFMxNy41MiwyLDEyLDJ6IE0xMiwyMGMtNC40MSwwLTgtMy41OS04LTggczMuNTktOCw4LThzOCwzLjU5LDgsOFMxNi40MSwyMCwxMiwyMHoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-cut: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkuNjQgNy42NGMuMjMtLjUuMzYtMS4wNS4zNi0xLjY0IDAtMi4yMS0xLjc5LTQtNC00UzIgMy43OSAyIDZzMS43OSA0IDQgNGMuNTkgMCAxLjE0LS4xMyAxLjY0LS4zNkwxMCAxMmwtMi4zNiAyLjM2QzcuMTQgMTQuMTMgNi41OSAxNCA2IDE0Yy0yLjIxIDAtNCAxLjc5LTQgNHMxLjc5IDQgNCA0IDQtMS43OSA0LTRjMC0uNTktLjEzLTEuMTQtLjM2LTEuNjRMMTIgMTRsNyA3aDN2LTFMOS42NCA3LjY0ek02IDhjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTAgMTJjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTYtNy41Yy0uMjggMC0uNS0uMjItLjUtLjVzLjIyLS41LjUtLjUuNS4yMi41LjUtLjIyLjUtLjUuNXpNMTkgM2wtNiA2IDIgMiA3LTdWM3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-delete: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2cHgiIGhlaWdodD0iMTZweCI+CiAgICA8cGF0aCBkPSJNMCAwaDI0djI0SDB6IiBmaWxsPSJub25lIiAvPgogICAgPHBhdGggY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjI2MjYyIiBkPSJNNiAxOWMwIDEuMS45IDIgMiAyaDhjMS4xIDAgMi0uOSAyLTJWN0g2djEyek0xOSA0aC0zLjVsLTEtMWgtNWwtMSAxSDV2MmgxNFY0eiIgLz4KPC9zdmc+Cg==);
  --jp-icon-download: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDloLTRWM0g5djZINWw3IDcgNy03ek01IDE4djJoMTR2LTJINXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-duplicate: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGNsaXAtcnVsZT0iZXZlbm9kZCIgZD0iTTIuNzk5OTggMC44NzVIOC44OTU4MkM5LjIwMDYxIDAuODc1IDkuNDQ5OTggMS4xMzkxNCA5LjQ0OTk4IDEuNDYxOThDOS40NDk5OCAxLjc4NDgyIDkuMjAwNjEgMi4wNDg5NiA4Ljg5NTgyIDIuMDQ4OTZIMy4zNTQxNUMzLjA0OTM2IDIuMDQ4OTYgMi43OTk5OCAyLjMxMzEgMi43OTk5OCAyLjYzNTk0VjkuNjc5NjlDMi43OTk5OCAxMC4wMDI1IDIuNTUwNjEgMTAuMjY2NyAyLjI0NTgyIDEwLjI2NjdDMS45NDEwMyAxMC4yNjY3IDEuNjkxNjUgMTAuMDAyNSAxLjY5MTY1IDkuNjc5NjlWMi4wNDg5NkMxLjY5MTY1IDEuNDAzMjggMi4xOTA0IDAuODc1IDIuNzk5OTggMC44NzVaTTUuMzY2NjUgMTEuOVY0LjU1SDExLjA4MzNWMTEuOUg1LjM2NjY1Wk00LjE0MTY1IDQuMTQxNjdDNC4xNDE2NSAzLjY5MDYzIDQuNTA3MjggMy4zMjUgNC45NTgzMiAzLjMyNUgxMS40OTE3QzExLjk0MjcgMy4zMjUgMTIuMzA4MyAzLjY5MDYzIDEyLjMwODMgNC4xNDE2N1YxMi4zMDgzQzEyLjMwODMgMTIuNzU5NCAxMS45NDI3IDEzLjEyNSAxMS40OTE3IDEzLjEyNUg0Ljk1ODMyQzQuNTA3MjggMTMuMTI1IDQuMTQxNjUgMTIuNzU5NCA0LjE0MTY1IDEyLjMwODNWNC4xNDE2N1oiIGZpbGw9IiM2MTYxNjEiLz4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBkPSJNOS40MzU3NCA4LjI2NTA3SDguMzY0MzFWOS4zMzY1QzguMzY0MzEgOS40NTQzNSA4LjI2Nzg4IDkuNTUwNzggOC4xNTAwMiA5LjU1MDc4QzguMDMyMTcgOS41NTA3OCA3LjkzNTc0IDkuNDU0MzUgNy45MzU3NCA5LjMzNjVWOC4yNjUwN0g2Ljg2NDMxQzYuNzQ2NDUgOC4yNjUwNyA2LjY1MDAyIDguMTY4NjQgNi42NTAwMiA4LjA1MDc4QzYuNjUwMDIgNy45MzI5MiA2Ljc0NjQ1IDcuODM2NSA2Ljg2NDMxIDcuODM2NUg3LjkzNTc0VjYuNzY1MDdDNy45MzU3NCA2LjY0NzIxIDguMDMyMTcgNi41NTA3OCA4LjE1MDAyIDYuNTUwNzhDOC4yNjc4OCA2LjU1MDc4IDguMzY0MzEgNi42NDcyMSA4LjM2NDMxIDYuNzY1MDdWNy44MzY1SDkuNDM1NzRDOS41NTM2IDcuODM2NSA5LjY1MDAyIDcuOTMyOTIgOS42NTAwMiA4LjA1MDc4QzkuNjUwMDIgOC4xNjg2NCA5LjU1MzYgOC4yNjUwNyA5LjQzNTc0IDguMjY1MDdaIiBmaWxsPSIjNjE2MTYxIiBzdHJva2U9IiM2MTYxNjEiIHN0cm9rZS13aWR0aD0iMC41Ii8+Cjwvc3ZnPgo=);
  --jp-icon-edit: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMgMTcuMjVWMjFoMy43NUwxNy44MSA5Ljk0bC0zLjc1LTMuNzVMMyAxNy4yNXpNMjAuNzEgNy4wNGMuMzktLjM5LjM5LTEuMDIgMC0xLjQxbC0yLjM0LTIuMzRjLS4zOS0uMzktMS4wMi0uMzktMS40MSAwbC0xLjgzIDEuODMgMy43NSAzLjc1IDEuODMtMS44M3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-ellipses: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iNSIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxOSIgY3k9IjEyIiByPSIyIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-error: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj48Y2lyY2xlIGN4PSIxMiIgY3k9IjE5IiByPSIyIi8+PHBhdGggZD0iTTEwIDNoNHYxMmgtNHoiLz48L2c+CjxwYXRoIGZpbGw9Im5vbmUiIGQ9Ik0wIDBoMjR2MjRIMHoiLz4KPC9zdmc+Cg==);
  --jp-icon-expand-all: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGgKICAgICAgICAgICAgZD0iTTggMmMxIDAgMTEgMCAxMiAwczIgMSAyIDJjMCAxIDAgMTEgMCAxMnMwIDItMiAyQzIwIDE0IDIwIDQgMjAgNFMxMCA0IDYgNGMwLTIgMS0yIDItMnoiIC8+CiAgICAgICAgPHBhdGgKICAgICAgICAgICAgZD0iTTE4IDhjMC0xLTEtMi0yLTJTNSA2IDQgNnMtMiAxLTIgMmMwIDEgMCAxMSAwIDEyczEgMiAyIDJjMSAwIDExIDAgMTIgMHMyLTEgMi0yYzAtMSAwLTExIDAtMTJ6bS0yIDB2MTJINFY4eiIgLz4KICAgICAgICA8cGF0aCBkPSJNMTEgMTBIOXYzSDZ2MmgzdjNoMnYtM2gzdi0yaC0zeiIgLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-extension: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwLjUgMTFIMTlWN2MwLTEuMS0uOS0yLTItMmgtNFYzLjVDMTMgMi4xMiAxMS44OCAxIDEwLjUgMVM4IDIuMTIgOCAzLjVWNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAydjMuOEgzLjVjMS40OSAwIDIuNyAxLjIxIDIuNyAyLjdzLTEuMjEgMi43LTIuNyAyLjdIMlYyMGMwIDEuMS45IDIgMiAyaDMuOHYtMS41YzAtMS40OSAxLjIxLTIuNyAyLjctMi43IDEuNDkgMCAyLjcgMS4yMSAyLjcgMi43VjIySDE3YzEuMSAwIDItLjkgMi0ydi00aDEuNWMxLjM4IDAgMi41LTEuMTIgMi41LTIuNVMyMS44OCAxMSAyMC41IDExeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-fast-forward: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTQgMThsOC41LTZMNCA2djEyem05LTEydjEybDguNS02TDEzIDZ6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-file-upload: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkgMTZoNnYtNmg0bC03LTctNyA3aDR6bS00IDJoMTR2Mkg1eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-file: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuMyA4LjJsLTUuNS01LjVjLS4zLS4zLS43LS41LTEuMi0uNUgzLjljLS44LjEtMS42LjktMS42IDEuOHYxNC4xYzAgLjkuNyAxLjYgMS42IDEuNmgxNC4yYy45IDAgMS42LS43IDEuNi0xLjZWOS40Yy4xLS41LS4xLS45LS40LTEuMnptLTUuOC0zLjNsMy40IDMuNmgtMy40VjQuOXptMy45IDEyLjdINC43Yy0uMSAwLS4yIDAtLjItLjJWNC43YzAtLjIuMS0uMy4yLS4zaDcuMnY0LjRzMCAuOC4zIDEuMWMuMy4zIDEuMS4zIDEuMS4zaDQuM3Y3LjJzLS4xLjItLjIuMnoiLz4KPC9zdmc+Cg==);
  --jp-icon-filter-dot: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTE0LDEyVjE5Ljg4QzE0LjA0LDIwLjE4IDEzLjk0LDIwLjUgMTMuNzEsMjAuNzFDMTMuMzIsMjEuMSAxMi42OSwyMS4xIDEyLjMsMjAuNzFMMTAuMjksMTguN0MxMC4wNiwxOC40NyA5Ljk2LDE4LjE2IDEwLDE3Ljg3VjEySDkuOTdMNC4yMSw0LjYyQzMuODcsNC4xOSAzLjk1LDMuNTYgNC4zOCwzLjIyQzQuNTcsMy4wOCA0Ljc4LDMgNSwzVjNIMTlWM0MxOS4yMiwzIDE5LjQzLDMuMDggMTkuNjIsMy4yMkMyMC4wNSwzLjU2IDIwLjEzLDQuMTkgMTkuNzksNC42MkwxNC4wMywxMkgxNFoiIC8+CiAgPC9nPgogIDxnIGNsYXNzPSJqcC1pY29uLWRvdCIgZmlsbD0iI0ZGRiI+CiAgICA8Y2lyY2xlIGN4PSIxOCIgY3k9IjE3IiByPSIzIj48L2NpcmNsZT4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-filter-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEwIDE4aDR2LTJoLTR2MnpNMyA2djJoMThWNkgzem0zIDdoMTJ2LTJINnYyeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-filter: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTE0LDEyVjE5Ljg4QzE0LjA0LDIwLjE4IDEzLjk0LDIwLjUgMTMuNzEsMjAuNzFDMTMuMzIsMjEuMSAxMi42OSwyMS4xIDEyLjMsMjAuNzFMMTAuMjksMTguN0MxMC4wNiwxOC40NyA5Ljk2LDE4LjE2IDEwLDE3Ljg3VjEySDkuOTdMNC4yMSw0LjYyQzMuODcsNC4xOSAzLjk1LDMuNTYgNC4zOCwzLjIyQzQuNTcsMy4wOCA0Ljc4LDMgNSwzVjNIMTlWM0MxOS4yMiwzIDE5LjQzLDMuMDggMTkuNjIsMy4yMkMyMC4wNSwzLjU2IDIwLjEzLDQuMTkgMTkuNzksNC42MkwxNC4wMywxMkgxNFoiIC8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-folder-favorite: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjRweCIgdmlld0JveD0iMCAwIDI0IDI0IiB3aWR0aD0iMjRweCIgZmlsbD0iIzAwMDAwMCI+CiAgPHBhdGggZD0iTTAgMGgyNHYyNEgwVjB6IiBmaWxsPSJub25lIi8+PHBhdGggY2xhc3M9ImpwLWljb24zIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzYxNjE2MSIgZD0iTTIwIDZoLThsLTItMkg0Yy0xLjEgMC0yIC45LTIgMnYxMmMwIDEuMS45IDIgMiAyaDE2YzEuMSAwIDItLjkgMi0yVjhjMC0xLjEtLjktMi0yLTJ6bS0yLjA2IDExTDE1IDE1LjI4IDEyLjA2IDE3bC43OC0zLjMzLTIuNTktMi4yNCAzLjQxLS4yOUwxNSA4bDEuMzQgMy4xNCAzLjQxLjI5LTIuNTkgMi4yNC43OCAzLjMzeiIvPgo8L3N2Zz4K);
  --jp-icon-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY4YzAtMS4xLS45LTItMi0yaC04bC0yLTJ6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-home: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjRweCIgdmlld0JveD0iMCAwIDI0IDI0IiB3aWR0aD0iMjRweCIgZmlsbD0iIzAwMDAwMCI+CiAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGNsYXNzPSJqcC1pY29uMyBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xMCAyMHYtNmg0djZoNXYtOGgzTDEyIDMgMiAxMmgzdjh6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-html5: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uMCBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMDAiIGQ9Ik0xMDguNCAwaDIzdjIyLjhoMjEuMlYwaDIzdjY5aC0yM1Y0NmgtMjF2MjNoLTIzLjJNMjA2IDIzaC0yMC4zVjBoNjMuN3YyM0gyMjl2NDZoLTIzbTUzLjUtNjloMjQuMWwxNC44IDI0LjNMMzEzLjIgMGgyNC4xdjY5aC0yM1YzNC44bC0xNi4xIDI0LjgtMTYuMS0yNC44VjY5aC0yMi42bTg5LjItNjloMjN2NDYuMmgzMi42VjY5aC01NS42Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI2U0NGQyNiIgZD0iTTEwNy42IDQ3MWwtMzMtMzcwLjRoMzYyLjhsLTMzIDM3MC4yTDI1NS43IDUxMiIvPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNmMTY1MjkiIGQ9Ik0yNTYgNDgwLjVWMTMxaDE0OC4zTDM3NiA0NDciLz4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNlYmViZWIiIGQ9Ik0xNDIgMTc2LjNoMTE0djQ1LjRoLTY0LjJsNC4yIDQ2LjVoNjB2NDUuM0gxNTQuNG0yIDIyLjhIMjAybDMuMiAzNi4zIDUwLjggMTMuNnY0Ny40bC05My4yLTI2Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIiBmaWxsPSIjZmZmIiBkPSJNMzY5LjYgMTc2LjNIMjU1Ljh2NDUuNGgxMDkuNm0tNC4xIDQ2LjVIMjU1Ljh2NDUuNGg1NmwtNS4zIDU5LTUwLjcgMTMuNnY0Ny4ybDkzLTI1LjgiLz4KPC9zdmc+Cg==);
  --jp-icon-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1icmFuZDQganAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNGRkYiIGQ9Ik0yLjIgMi4yaDE3LjV2MTcuNUgyLjJ6Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzNGNTFCNSIgZD0iTTIuMiAyLjJ2MTcuNWgxNy41bC4xLTE3LjVIMi4yem0xMi4xIDIuMmMxLjIgMCAyLjIgMSAyLjIgMi4ycy0xIDIuMi0yLjIgMi4yLTIuMi0xLTIuMi0yLjIgMS0yLjIgMi4yLTIuMnpNNC40IDE3LjZsMy4zLTguOCAzLjMgNi42IDIuMi0zLjIgNC40IDUuNEg0LjR6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-info: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUwLjk3OCA1MC45NzgiPgoJPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KCQk8cGF0aCBkPSJNNDMuNTIsNy40NThDMzguNzExLDIuNjQ4LDMyLjMwNywwLDI1LjQ4OSwwQzE4LjY3LDAsMTIuMjY2LDIuNjQ4LDcuNDU4LDcuNDU4CgkJCWMtOS45NDMsOS45NDEtOS45NDMsMjYuMTE5LDAsMzYuMDYyYzQuODA5LDQuODA5LDExLjIxMiw3LjQ1NiwxOC4wMzEsNy40NThjMCwwLDAuMDAxLDAsMC4wMDIsMAoJCQljNi44MTYsMCwxMy4yMjEtMi42NDgsMTguMDI5LTcuNDU4YzQuODA5LTQuODA5LDcuNDU3LTExLjIxMiw3LjQ1Ny0xOC4wM0M1MC45NzcsMTguNjcsNDguMzI4LDEyLjI2Niw0My41Miw3LjQ1OHoKCQkJIE00Mi4xMDYsNDIuMTA1Yy00LjQzMiw0LjQzMS0xMC4zMzIsNi44NzItMTYuNjE1LDYuODcyaC0wLjAwMmMtNi4yODUtMC4wMDEtMTIuMTg3LTIuNDQxLTE2LjYxNy02Ljg3MgoJCQljLTkuMTYyLTkuMTYzLTkuMTYyLTI0LjA3MSwwLTMzLjIzM0MxMy4zMDMsNC40NCwxOS4yMDQsMiwyNS40ODksMmM2LjI4NCwwLDEyLjE4NiwyLjQ0LDE2LjYxNyw2Ljg3MgoJCQljNC40MzEsNC40MzEsNi44NzEsMTAuMzMyLDYuODcxLDE2LjYxN0M0OC45NzcsMzEuNzcyLDQ2LjUzNiwzNy42NzUsNDIuMTA2LDQyLjEwNXoiLz4KCQk8cGF0aCBkPSJNMjMuNTc4LDMyLjIxOGMtMC4wMjMtMS43MzQsMC4xNDMtMy4wNTksMC40OTYtMy45NzJjMC4zNTMtMC45MTMsMS4xMS0xLjk5NywyLjI3Mi0zLjI1MwoJCQljMC40NjgtMC41MzYsMC45MjMtMS4wNjIsMS4zNjctMS41NzVjMC42MjYtMC43NTMsMS4xMDQtMS40NzgsMS40MzYtMi4xNzVjMC4zMzEtMC43MDcsMC40OTUtMS41NDEsMC40OTUtMi41CgkJCWMwLTEuMDk2LTAuMjYtMi4wODgtMC43NzktMi45NzljLTAuNTY1LTAuODc5LTEuNTAxLTEuMzM2LTIuODA2LTEuMzY5Yy0xLjgwMiwwLjA1Ny0yLjk4NSwwLjY2Ny0zLjU1LDEuODMyCgkJCWMtMC4zMDEsMC41MzUtMC41MDMsMS4xNDEtMC42MDcsMS44MTRjLTAuMTM5LDAuNzA3LTAuMjA3LDEuNDMyLTAuMjA3LDIuMTc0aC0yLjkzN2MtMC4wOTEtMi4yMDgsMC40MDctNC4xMTQsMS40OTMtNS43MTkKCQkJYzEuMDYyLTEuNjQsMi44NTUtMi40ODEsNS4zNzgtMi41MjdjMi4xNiwwLjAyMywzLjg3NCwwLjYwOCw1LjE0MSwxLjc1OGMxLjI3OCwxLjE2LDEuOTI5LDIuNzY0LDEuOTUsNC44MTEKCQkJYzAsMS4xNDItMC4xMzcsMi4xMTEtMC40MSwyLjkxMWMtMC4zMDksMC44NDUtMC43MzEsMS41OTMtMS4yNjgsMi4yNDNjLTAuNDkyLDAuNjUtMS4wNjgsMS4zMTgtMS43MywyLjAwMgoJCQljLTAuNjUsMC42OTctMS4zMTMsMS40NzktMS45ODcsMi4zNDZjLTAuMjM5LDAuMzc3LTAuNDI5LDAuNzc3LTAuNTY1LDEuMTk5Yy0wLjE2LDAuOTU5LTAuMjE3LDEuOTUxLTAuMTcxLDIuOTc5CgkJCUMyNi41ODksMzIuMjE4LDIzLjU3OCwzMi4yMTgsMjMuNTc4LDMyLjIxOHogTTIzLjU3OCwzOC4yMnYtMy40ODRoMy4wNzZ2My40ODRIMjMuNTc4eiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-inspector: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaW5zcGVjdG9yLWljb24tY29sb3IganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY2YzAtMS4xLS45LTItMi0yem0tNSAxNEg0di00aDExdjR6bTAtNUg0VjloMTF2NHptNSA1aC00VjloNHY5eiIvPgo8L3N2Zz4K);
  --jp-icon-json: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtanNvbi1pY29uLWNvbG9yIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI0Y5QTgyNSI+CiAgICA8cGF0aCBkPSJNMjAuMiAxMS44Yy0xLjYgMC0xLjcuNS0xLjcgMSAwIC40LjEuOS4xIDEuMy4xLjUuMS45LjEgMS4zIDAgMS43LTEuNCAyLjMtMy41IDIuM2gtLjl2LTEuOWguNWMxLjEgMCAxLjQgMCAxLjQtLjggMC0uMyAwLS42LS4xLTEgMC0uNC0uMS0uOC0uMS0xLjIgMC0xLjMgMC0xLjggMS4zLTItMS4zLS4yLTEuMy0uNy0xLjMtMiAwLS40LjEtLjguMS0xLjIuMS0uNC4xLS43LjEtMSAwLS44LS40LS43LTEuNC0uOGgtLjVWNC4xaC45YzIuMiAwIDMuNS43IDMuNSAyLjMgMCAuNC0uMS45LS4xIDEuMy0uMS41LS4xLjktLjEgMS4zIDAgLjUuMiAxIDEuNyAxdjEuOHpNMS44IDEwLjFjMS42IDAgMS43LS41IDEuNy0xIDAtLjQtLjEtLjktLjEtMS4zLS4xLS41LS4xLS45LS4xLTEuMyAwLTEuNiAxLjQtMi4zIDMuNS0yLjNoLjl2MS45aC0uNWMtMSAwLTEuNCAwLTEuNC44IDAgLjMgMCAuNi4xIDEgMCAuMi4xLjYuMSAxIDAgMS4zIDAgMS44LTEuMyAyQzYgMTEuMiA2IDExLjcgNiAxM2MwIC40LS4xLjgtLjEgMS4yLS4xLjMtLjEuNy0uMSAxIDAgLjguMy44IDEuNC44aC41djEuOWgtLjljLTIuMSAwLTMuNS0uNi0zLjUtMi4zIDAtLjQuMS0uOS4xLTEuMy4xLS41LjEtLjkuMS0xLjMgMC0uNS0uMi0xLTEuNy0xdi0xLjl6Ii8+CiAgICA8Y2lyY2xlIGN4PSIxMSIgY3k9IjEzLjgiIHI9IjIuMSIvPgogICAgPGNpcmNsZSBjeD0iMTEiIGN5PSI4LjIiIHI9IjIuMSIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-julia: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDMyNSAzMDAiPgogIDxnIGNsYXNzPSJqcC1icmFuZDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjY2IzYzMzIj4KICAgIDxwYXRoIGQ9Ik0gMTUwLjg5ODQzOCAyMjUgQyAxNTAuODk4NDM4IDI2Ni40MjE4NzUgMTE3LjMyMDMxMiAzMDAgNzUuODk4NDM4IDMwMCBDIDM0LjQ3NjU2MiAzMDAgMC44OTg0MzggMjY2LjQyMTg3NSAwLjg5ODQzOCAyMjUgQyAwLjg5ODQzOCAxODMuNTc4MTI1IDM0LjQ3NjU2MiAxNTAgNzUuODk4NDM4IDE1MCBDIDExNy4zMjAzMTIgMTUwIDE1MC44OTg0MzggMTgzLjU3ODEyNSAxNTAuODk4NDM4IDIyNSIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzM4OTgyNiI+CiAgICA8cGF0aCBkPSJNIDIzNy41IDc1IEMgMjM3LjUgMTE2LjQyMTg3NSAyMDMuOTIxODc1IDE1MCAxNjIuNSAxNTAgQyAxMjEuMDc4MTI1IDE1MCA4Ny41IDExNi40MjE4NzUgODcuNSA3NSBDIDg3LjUgMzMuNTc4MTI1IDEyMS4wNzgxMjUgMCAxNjIuNSAwIEMgMjAzLjkyMTg3NSAwIDIzNy41IDMzLjU3ODEyNSAyMzcuNSA3NSIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzk1NThiMiI+CiAgICA8cGF0aCBkPSJNIDMyNC4xMDE1NjIgMjI1IEMgMzI0LjEwMTU2MiAyNjYuNDIxODc1IDI5MC41MjM0MzggMzAwIDI0OS4xMDE1NjIgMzAwIEMgMjA3LjY3OTY4OCAzMDAgMTc0LjEwMTU2MiAyNjYuNDIxODc1IDE3NC4xMDE1NjIgMjI1IEMgMTc0LjEwMTU2MiAxODMuNTc4MTI1IDIwNy42Nzk2ODggMTUwIDI0OS4xMDE1NjIgMTUwIEMgMjkwLjUyMzQzOCAxNTAgMzI0LjEwMTU2MiAxODMuNTc4MTI1IDMyNC4xMDE1NjIgMjI1Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-jupyter-favicon: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTUyIiBoZWlnaHQ9IjE2NSIgdmlld0JveD0iMCAwIDE1MiAxNjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgPGcgY2xhc3M9ImpwLWp1cHl0ZXItaWNvbi1jb2xvciIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA3ODk0NywgMTEwLjU4MjkyNykiIGQ9Ik03NS45NDIyODQyLDI5LjU4MDQ1NjEgQzQzLjMwMjM5NDcsMjkuNTgwNDU2MSAxNC43OTY3ODMyLDE3LjY1MzQ2MzQgMCwwIEM1LjUxMDgzMjExLDE1Ljg0MDY4MjkgMTUuNzgxNTM4OSwyOS41NjY3NzMyIDI5LjM5MDQ5NDcsMzkuMjc4NDE3MSBDNDIuOTk5Nyw0OC45ODk4NTM3IDU5LjI3MzcsNTQuMjA2NzgwNSA3NS45NjA1Nzg5LDU0LjIwNjc4MDUgQzkyLjY0NzQ1NzksNTQuMjA2NzgwNSAxMDguOTIxNDU4LDQ4Ljk4OTg1MzcgMTIyLjUzMDY2MywzOS4yNzg0MTcxIEMxMzYuMTM5NDUzLDI5LjU2Njc3MzIgMTQ2LjQxMDI4NCwxNS44NDA2ODI5IDE1MS45MjExNTgsMCBDMTM3LjA4Nzg2OCwxNy42NTM0NjM0IDEwOC41ODI1ODksMjkuNTgwNDU2MSA3NS45NDIyODQyLDI5LjU4MDQ1NjEgTDc1Ljk0MjI4NDIsMjkuNTgwNDU2MSBaIiAvPgogICAgPHBhdGggdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMzczNjgsIDAuNzA0ODc4KSIgZD0iTTc1Ljk3ODQ1NzksMjQuNjI2NDA3MyBDMTA4LjYxODc2MywyNC42MjY0MDczIDEzNy4xMjQ0NTgsMzYuNTUzNDQxNSAxNTEuOTIxMTU4LDU0LjIwNjc4MDUgQzE0Ni40MTAyODQsMzguMzY2MjIyIDEzNi4xMzk0NTMsMjQuNjQwMTMxNyAxMjIuNTMwNjYzLDE0LjkyODQ4NzggQzEwOC45MjE0NTgsNS4yMTY4NDM5IDkyLjY0NzQ1NzksMCA3NS45NjA1Nzg5LDAgQzU5LjI3MzcsMCA0Mi45OTk3LDUuMjE2ODQzOSAyOS4zOTA0OTQ3LDE0LjkyODQ4NzggQzE1Ljc4MTUzODksMjQuNjQwMTMxNyA1LjUxMDgzMjExLDM4LjM2NjIyMiAwLDU0LjIwNjc4MDUgQzE0LjgzMzA4MTYsMzYuNTg5OTI5MyA0My4zMzg1Njg0LDI0LjYyNjQwNzMgNzUuOTc4NDU3OSwyNC42MjY0MDczIEw3NS45Nzg0NTc5LDI0LjYyNjQwNzMgWiIgLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-jupyter: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzkiIGhlaWdodD0iNTEiIHZpZXdCb3g9IjAgMCAzOSA1MSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtMTYzOCAtMjI4MSkiPgogICAgIDxnIGNsYXNzPSJqcC1qdXB5dGVyLWljb24tY29sb3IiIGZpbGw9IiNGMzc3MjYiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5Ljc0IDIzMTEuOTgpIiBkPSJNIDE4LjI2NDYgNy4xMzQxMUMgMTAuNDE0NSA3LjEzNDExIDMuNTU4NzIgNC4yNTc2IDAgMEMgMS4zMjUzOSAzLjgyMDQgMy43OTU1NiA3LjEzMDgxIDcuMDY4NiA5LjQ3MzAzQyAxMC4zNDE3IDExLjgxNTIgMTQuMjU1NyAxMy4wNzM0IDE4LjI2OSAxMy4wNzM0QyAyMi4yODIzIDEzLjA3MzQgMjYuMTk2MyAxMS44MTUyIDI5LjQ2OTQgOS40NzMwM0MgMzIuNzQyNCA3LjEzMDgxIDM1LjIxMjYgMy44MjA0IDM2LjUzOCAwQyAzMi45NzA1IDQuMjU3NiAyNi4xMTQ4IDcuMTM0MTEgMTguMjY0NiA3LjEzNDExWiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5LjczIDIyODUuNDgpIiBkPSJNIDE4LjI3MzMgNS45MzkzMUMgMjYuMTIzNSA1LjkzOTMxIDMyLjk3OTMgOC44MTU4MyAzNi41MzggMTMuMDczNEMgMzUuMjEyNiA5LjI1MzAzIDMyLjc0MjQgNS45NDI2MiAyOS40Njk0IDMuNjAwNEMgMjYuMTk2MyAxLjI1ODE4IDIyLjI4MjMgMCAxOC4yNjkgMEMgMTQuMjU1NyAwIDEwLjM0MTcgMS4yNTgxOCA3LjA2ODYgMy42MDA0QyAzLjc5NTU2IDUuOTQyNjIgMS4zMjUzOSA5LjI1MzAzIDAgMTMuMDczNEMgMy41Njc0NSA4LjgyNDYzIDEwLjQyMzIgNS45MzkzMSAxOC4yNzMzIDUuOTM5MzFaIi8+CiAgICA8L2c+CiAgICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjY5LjMgMjI4MS4zMSkiIGQ9Ik0gNS44OTM1MyAyLjg0NEMgNS45MTg4OSAzLjQzMTY1IDUuNzcwODUgNC4wMTM2NyA1LjQ2ODE1IDQuNTE2NDVDIDUuMTY1NDUgNS4wMTkyMiA0LjcyMTY4IDUuNDIwMTUgNC4xOTI5OSA1LjY2ODUxQyAzLjY2NDMgNS45MTY4OCAzLjA3NDQ0IDYuMDAxNTEgMi40OTgwNSA1LjkxMTcxQyAxLjkyMTY2IDUuODIxOSAxLjM4NDYzIDUuNTYxNyAwLjk1NDg5OCA1LjE2NDAxQyAwLjUyNTE3IDQuNzY2MzMgMC4yMjIwNTYgNC4yNDkwMyAwLjA4MzkwMzcgMy42Nzc1N0MgLTAuMDU0MjQ4MyAzLjEwNjExIC0wLjAyMTIzIDIuNTA2MTcgMC4xNzg3ODEgMS45NTM2NEMgMC4zNzg3OTMgMS40MDExIDAuNzM2ODA5IDAuOTIwODE3IDEuMjA3NTQgMC41NzM1MzhDIDEuNjc4MjYgMC4yMjYyNTkgMi4yNDA1NSAwLjAyNzU5MTkgMi44MjMyNiAwLjAwMjY3MjI5QyAzLjYwMzg5IC0wLjAzMDcxMTUgNC4zNjU3MyAwLjI0OTc4OSA0Ljk0MTQyIDAuNzgyNTUxQyA1LjUxNzExIDEuMzE1MzEgNS44NTk1NiAyLjA1Njc2IDUuODkzNTMgMi44NDRaIi8+CiAgICAgIDxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE2MzkuOCAyMzIzLjgxKSIgZD0iTSA3LjQyNzg5IDMuNTgzMzhDIDcuNDYwMDggNC4zMjQzIDcuMjczNTUgNS4wNTgxOSA2Ljg5MTkzIDUuNjkyMTNDIDYuNTEwMzEgNi4zMjYwNyA1Ljk1MDc1IDYuODMxNTYgNS4yODQxMSA3LjE0NDZDIDQuNjE3NDcgNy40NTc2MyAzLjg3MzcxIDcuNTY0MTQgMy4xNDcwMiA3LjQ1MDYzQyAyLjQyMDMyIDcuMzM3MTIgMS43NDMzNiA3LjAwODcgMS4yMDE4NCA2LjUwNjk1QyAwLjY2MDMyOCA2LjAwNTIgMC4yNzg2MSA1LjM1MjY4IDAuMTA1MDE3IDQuNjMyMDJDIC0wLjA2ODU3NTcgMy45MTEzNSAtMC4wMjYyMzYxIDMuMTU0OTQgMC4yMjY2NzUgMi40NTg1NkMgMC40Nzk1ODcgMS43NjIxNyAwLjkzMTY5NyAxLjE1NzEzIDEuNTI1NzYgMC43MjAwMzNDIDIuMTE5ODMgMC4yODI5MzUgMi44MjkxNCAwLjAzMzQzOTUgMy41NjM4OSAwLjAwMzEzMzQ0QyA0LjU0NjY3IC0wLjAzNzQwMzMgNS41MDUyOSAwLjMxNjcwNiA2LjIyOTYxIDAuOTg3ODM1QyA2Ljk1MzkzIDEuNjU4OTYgNy4zODQ4NCAyLjU5MjM1IDcuNDI3ODkgMy41ODMzOEwgNy40Mjc4OSAzLjU4MzM4WiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM4LjM2IDIyODYuMDYpIiBkPSJNIDIuMjc0NzEgNC4zOTYyOUMgMS44NDM2MyA0LjQxNTA4IDEuNDE2NzEgNC4zMDQ0NSAxLjA0Nzk5IDQuMDc4NDNDIDAuNjc5MjY4IDMuODUyNCAwLjM4NTMyOCAzLjUyMTE0IDAuMjAzMzcxIDMuMTI2NTZDIDAuMDIxNDEzNiAyLjczMTk4IC0wLjA0MDM3OTggMi4yOTE4MyAwLjAyNTgxMTYgMS44NjE4MUMgMC4wOTIwMDMxIDEuNDMxOCAwLjI4MzIwNCAxLjAzMTI2IDAuNTc1MjEzIDAuNzEwODgzQyAwLjg2NzIyMiAwLjM5MDUxIDEuMjQ2OTEgMC4xNjQ3MDggMS42NjYyMiAwLjA2MjA1OTJDIDIuMDg1NTMgLTAuMDQwNTg5NyAyLjUyNTYxIC0wLjAxNTQ3MTQgMi45MzA3NiAwLjEzNDIzNUMgMy4zMzU5MSAwLjI4Mzk0MSAzLjY4NzkyIDAuNTUxNTA1IDMuOTQyMjIgMC45MDMwNkMgNC4xOTY1MiAxLjI1NDYyIDQuMzQxNjkgMS42NzQzNiA0LjM1OTM1IDIuMTA5MTZDIDQuMzgyOTkgMi42OTEwNyA0LjE3Njc4IDMuMjU4NjkgMy43ODU5NyAzLjY4NzQ2QyAzLjM5NTE2IDQuMTE2MjQgMi44NTE2NiA0LjM3MTE2IDIuMjc0NzEgNC4zOTYyOUwgMi4yNzQ3MSA0LjM5NjI5WiIvPgogICAgPC9nPgogIDwvZz4+Cjwvc3ZnPgo=);
  --jp-icon-jupyterlab-wordmark: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMDAiIHZpZXdCb3g9IjAgMCAxODYwLjggNDc1Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0RTRFNEUiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ4MC4xMzY0MDEsIDY0LjI3MTQ5MykiPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDU4Ljg3NTU2NikiPgogICAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA4NzYwMywgMC4xNDAyOTQpIj4KICAgICAgICA8cGF0aCBkPSJNLTQyNi45LDE2OS44YzAsNDguNy0zLjcsNjQuNy0xMy42LDc2LjRjLTEwLjgsMTAtMjUsMTUuNS0zOS43LDE1LjVsMy43LDI5IGMyMi44LDAuMyw0NC44LTcuOSw2MS45LTIzLjFjMTcuOC0xOC41LDI0LTQ0LjEsMjQtODMuM1YwSC00Mjd2MTcwLjFMLTQyNi45LDE2OS44TC00MjYuOSwxNjkuOHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTU1LjA0NTI5NiwgNTYuODM3MTA0KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTYyNDUzLCAxLjc5OTg0MikiPgogICAgICAgIDxwYXRoIGQ9Ik0tMzEyLDE0OGMwLDIxLDAsMzkuNSwxLjcsNTUuNGgtMzEuOGwtMi4xLTMzLjNoLTAuOGMtNi43LDExLjYtMTYuNCwyMS4zLTI4LDI3LjkgYy0xMS42LDYuNi0yNC44LDEwLTM4LjIsOS44Yy0zMS40LDAtNjktMTcuNy02OS04OVYwaDM2LjR2MTEyLjdjMCwzOC43LDExLjYsNjQuNyw0NC42LDY0LjdjMTAuMy0wLjIsMjAuNC0zLjUsMjguOS05LjQgYzguNS01LjksMTUuMS0xNC4zLDE4LjktMjMuOWMyLjItNi4xLDMuMy0xMi41LDMuMy0xOC45VjAuMmgzNi40VjE0OEgtMzEyTC0zMTIsMTQ4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzOTAuMDEzMzIyLCA1My40Nzk2MzgpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS43MDY0NTgsIDAuMjMxNDI1KSI+CiAgICAgICAgPHBhdGggZD0iTS00NzguNiw3MS40YzAtMjYtMC44LTQ3LTEuNy02Ni43aDMyLjdsMS43LDM0LjhoMC44YzcuMS0xMi41LDE3LjUtMjIuOCwzMC4xLTI5LjcgYzEyLjUtNywyNi43LTEwLjMsNDEtOS44YzQ4LjMsMCw4NC43LDQxLjcsODQuNywxMDMuM2MwLDczLjEtNDMuNywxMDkuMi05MSwxMDkuMmMtMTIuMSwwLjUtMjQuMi0yLjItMzUtNy44IGMtMTAuOC01LjYtMTkuOS0xMy45LTI2LjYtMjQuMmgtMC44VjI5MWgtMzZ2LTIyMEwtNDc4LjYsNzEuNEwtNDc4LjYsNzEuNHogTS00NDIuNiwxMjUuNmMwLjEsNS4xLDAuNiwxMC4xLDEuNywxNS4xIGMzLDEyLjMsOS45LDIzLjMsMTkuOCwzMS4xYzkuOSw3LjgsMjIuMSwxMi4xLDM0LjcsMTIuMWMzOC41LDAsNjAuNy0zMS45LDYwLjctNzguNWMwLTQwLjctMjEuMS03NS42LTU5LjUtNzUuNiBjLTEyLjksMC40LTI1LjMsNS4xLTM1LjMsMTMuNGMtOS45LDguMy0xNi45LDE5LjctMTkuNiwzMi40Yy0xLjUsNC45LTIuMywxMC0yLjUsMTUuMVYxMjUuNkwtNDQyLjYsMTI1LjZMLTQ0Mi42LDEyNS42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg2MDYuNzQwNzI2LCA1Ni44MzcxMDQpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC43NTEyMjYsIDEuOTg5Mjk5KSI+CiAgICAgICAgPHBhdGggZD0iTS00NDAuOCwwbDQzLjcsMTIwLjFjNC41LDEzLjQsOS41LDI5LjQsMTIuOCw0MS43aDAuOGMzLjctMTIuMiw3LjktMjcuNywxMi44LTQyLjQgbDM5LjctMTE5LjJoMzguNUwtMzQ2LjksMTQ1Yy0yNiw2OS43LTQzLjcsMTA1LjQtNjguNiwxMjcuMmMtMTIuNSwxMS43LTI3LjksMjAtNDQuNiwyMy45bC05LjEtMzEuMSBjMTEuNy0zLjksMjIuNS0xMC4xLDMxLjgtMTguMWMxMy4yLTExLjEsMjMuNy0yNS4yLDMwLjYtNDEuMmMxLjUtMi44LDIuNS01LjcsMi45LTguOGMtMC4zLTMuMy0xLjItNi42LTIuNS05LjdMLTQ4MC4yLDAuMSBoMzkuN0wtNDQwLjgsMEwtNDQwLjgsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoODIyLjc0ODEwNCwgMC4wMDAwMDApIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS40NjQwNTAsIDAuMzc4OTE0KSI+CiAgICAgICAgPHBhdGggZD0iTS00MTMuNywwdjU4LjNoNTJ2MjguMmgtNTJWMTk2YzAsMjUsNywzOS41LDI3LjMsMzkuNWM3LjEsMC4xLDE0LjItMC43LDIxLjEtMi41IGwxLjcsMjcuN2MtMTAuMywzLjctMjEuMyw1LjQtMzIuMiw1Yy03LjMsMC40LTE0LjYtMC43LTIxLjMtMy40Yy02LjgtMi43LTEyLjktNi44LTE3LjktMTIuMWMtMTAuMy0xMC45LTE0LjEtMjktMTQuMS01Mi45IFY4Ni41aC0zMVY1OC4zaDMxVjkuNkwtNDEzLjcsMEwtNDEzLjcsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOTc0LjQzMzI4NiwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuOTkwMDM0LCAwLjYxMDMzOSkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDQ1LjgsMTEzYzAuOCw1MCwzMi4yLDcwLjYsNjguNiw3MC42YzE5LDAuNiwzNy45LTMsNTUuMy0xMC41bDYuMiwyNi40IGMtMjAuOSw4LjktNDMuNSwxMy4xLTY2LjIsMTIuNmMtNjEuNSwwLTk4LjMtNDEuMi05OC4zLTEwMi41Qy00ODAuMiw0OC4yLTQ0NC43LDAtMzg2LjUsMGM2NS4yLDAsODIuNyw1OC4zLDgyLjcsOTUuNyBjLTAuMSw1LjgtMC41LDExLjUtMS4yLDE3LjJoLTE0MC42SC00NDUuOEwtNDQ1LjgsMTEzeiBNLTMzOS4yLDg2LjZjMC40LTIzLjUtOS41LTYwLjEtNTAuNC02MC4xIGMtMzYuOCwwLTUyLjgsMzQuNC01NS43LDYwLjFILTMzOS4yTC0zMzkuMiw4Ni42TC0zMzkuMiw4Ni42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjAxLjk2MTA1OCwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuMTc5NjQwLCAwLjcwNTA2OCkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDc4LjYsNjhjMC0yMy45LTAuNC00NC41LTEuNy02My40aDMxLjhsMS4yLDM5LjloMS43YzkuMS0yNy4zLDMxLTQ0LjUsNTUuMy00NC41IGMzLjUtMC4xLDcsMC40LDEwLjMsMS4ydjM0LjhjLTQuMS0wLjktOC4yLTEuMy0xMi40LTEuMmMtMjUuNiwwLTQzLjcsMTkuNy00OC43LDQ3LjRjLTEsNS43LTEuNiwxMS41LTEuNywxNy4ydjEwOC4zaC0zNlY2OCBMLTQ3OC42LDY4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCBkPSJNMTM1Mi4zLDMyNi4yaDM3VjI4aC0zN1YzMjYuMnogTTE2MDQuOCwzMjYuMmMtMi41LTEzLjktMy40LTMxLjEtMy40LTQ4Ljd2LTc2IGMwLTQwLjctMTUuMS04My4xLTc3LjMtODMuMWMtMjUuNiwwLTUwLDcuMS02Ni44LDE4LjFsOC40LDI0LjRjMTQuMy05LjIsMzQtMTUuMSw1My0xNS4xYzQxLjYsMCw0Ni4yLDMwLjIsNDYuMiw0N3Y0LjIgYy03OC42LTAuNC0xMjIuMywyNi41LTEyMi4zLDc1LjZjMCwyOS40LDIxLDU4LjQsNjIuMiw1OC40YzI5LDAsNTAuOS0xNC4zLDYyLjItMzAuMmgxLjNsMi45LDI1LjZIMTYwNC44eiBNMTU2NS43LDI1Ny43IGMwLDMuOC0wLjgsOC0yLjEsMTEuOGMtNS45LDE3LjItMjIuNywzNC00OS4yLDM0Yy0xOC45LDAtMzQuOS0xMS4zLTM0LjktMzUuM2MwLTM5LjUsNDUuOC00Ni42LDg2LjItNDUuOFYyNTcuN3ogTTE2OTguNSwzMjYuMiBsMS43LTMzLjZoMS4zYzE1LjEsMjYuOSwzOC43LDM4LjIsNjguMSwzOC4yYzQ1LjQsMCw5MS4yLTM2LjEsOTEuMi0xMDguOGMwLjQtNjEuNy0zNS4zLTEwMy43LTg1LjctMTAzLjcgYy0zMi44LDAtNTYuMywxNC43LTY5LjMsMzcuNGgtMC44VjI4aC0zNi42djI0NS43YzAsMTguMS0wLjgsMzguNi0xLjcsNTIuNUgxNjk4LjV6IE0xNzA0LjgsMjA4LjJjMC01LjksMS4zLTEwLjksMi4xLTE1LjEgYzcuNi0yOC4xLDMxLjEtNDUuNCw1Ni4zLTQ1LjRjMzkuNSwwLDYwLjUsMzQuOSw2MC41LDc1LjZjMCw0Ni42LTIzLjEsNzguMS02MS44LDc4LjFjLTI2LjksMC00OC4zLTE3LjYtNTUuNS00My4zIGMtMC44LTQuMi0xLjctOC44LTEuNy0xMy40VjIwOC4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgZmlsbD0iIzYxNjE2MSIgZD0iTTE1IDlIOXY2aDZWOXptLTIgNGgtMnYtMmgydjJ6bTgtMlY5aC0yVjdjMC0xLjEtLjktMi0yLTJoLTJWM2gtMnYyaC0yVjNIOXYySDdjLTEuMSAwLTIgLjktMiAydjJIM3YyaDJ2MkgzdjJoMnYyYzAgMS4xLjkgMiAyIDJoMnYyaDJ2LTJoMnYyaDJ2LTJoMmMxLjEgMCAyLS45IDItMnYtMmgydi0yaC0ydi0yaDJ6bS00IDZIN1Y3aDEwdjEweiIvPgo8L3N2Zz4K);
  --jp-icon-keyboard: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMTdjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY3YzAtMS4xLS45LTItMi0yem0tOSAzaDJ2MmgtMlY4em0wIDNoMnYyaC0ydi0yek04IDhoMnYySDhWOHptMCAzaDJ2Mkg4di0yem0tMSAySDV2LTJoMnYyem0wLTNINVY4aDJ2MnptOSA3SDh2LTJoOHYyem0wLTRoLTJ2LTJoMnYyem0wLTNoLTJWOGgydjJ6bTMgM2gtMnYtMmgydjJ6bTAtM2gtMlY4aDJ2MnoiLz4KPC9zdmc+Cg==);
  --jp-icon-launch: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMzIgMzIiIHdpZHRoPSIzMiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0yNiwyOEg2YTIuMDAyNywyLjAwMjcsMCwwLDEtMi0yVjZBMi4wMDI3LDIuMDAyNywwLDAsMSw2LDRIMTZWNkg2VjI2SDI2VjE2aDJWMjZBMi4wMDI3LDIuMDAyNywwLDAsMSwyNiwyOFoiLz4KICAgIDxwb2x5Z29uIHBvaW50cz0iMjAgMiAyMCA0IDI2LjU4NiA0IDE4IDEyLjU4NiAxOS40MTQgMTQgMjggNS40MTQgMjggMTIgMzAgMTIgMzAgMiAyMCAyIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-launcher: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkgMTlINVY1aDdWM0g1YTIgMiAwIDAwLTIgMnYxNGEyIDIgMCAwMDIgMmgxNGMxLjEgMCAyLS45IDItMnYtN2gtMnY3ek0xNCAzdjJoMy41OWwtOS44MyA5LjgzIDEuNDEgMS40MUwxOSA2LjQxVjEwaDJWM2gtN3oiLz4KPC9zdmc+Cg==);
  --jp-icon-line-form: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGZpbGw9IndoaXRlIiBkPSJNNS44OCA0LjEyTDEzLjc2IDEybC03Ljg4IDcuODhMOCAyMmwxMC0xMEw4IDJ6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-link: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMuOSAxMmMwLTEuNzEgMS4zOS0zLjEgMy4xLTMuMWg0VjdIN2MtMi43NiAwLTUgMi4yNC01IDVzMi4yNCA1IDUgNWg0di0xLjlIN2MtMS43MSAwLTMuMS0xLjM5LTMuMS0zLjF6TTggMTNoOHYtMkg4djJ6bTktNmgtNHYxLjloNGMxLjcxIDAgMy4xIDEuMzkgMy4xIDMuMXMtMS4zOSAzLjEtMy4xIDMuMWgtNFYxN2g0YzIuNzYgMCA1LTIuMjQgNS01cy0yLjI0LTUtNS01eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xOSA1djE0SDVWNWgxNG0xLjEtMkgzLjljLS41IDAtLjkuNC0uOS45djE2LjJjMCAuNC40LjkuOS45aDE2LjJjLjQgMCAuOS0uNS45LS45VjMuOWMwLS41LS41LS45LS45LS45ek0xMSA3aDZ2MmgtNlY3em0wIDRoNnYyaC02di0yem0wIDRoNnYyaC02ek03IDdoMnYySDd6bTAgNGgydjJIN3ptMCA0aDJ2Mkg3eiIvPgo8L3N2Zz4K);
  --jp-icon-markdown: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjN0IxRkEyIiBkPSJNNSAxNC45aDEybC02LjEgNnptOS40LTYuOGMwLTEuMy0uMS0yLjktLjEtNC41LS40IDEuNC0uOSAyLjktMS4zIDQuM2wtMS4zIDQuM2gtMkw4LjUgNy45Yy0uNC0xLjMtLjctMi45LTEtNC4zLS4xIDEuNi0uMSAzLjItLjIgNC42TDcgMTIuNEg0LjhsLjctMTFoMy4zTDEwIDVjLjQgMS4yLjcgMi43IDEgMy45LjMtMS4yLjctMi42IDEtMy45bDEuMi0zLjdoMy4zbC42IDExaC0yLjRsLS4zLTQuMnoiLz4KPC9zdmc+Cg==);
  --jp-icon-move-down: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBkPSJNMTIuNDcxIDcuNTI4OTlDMTIuNzYzMiA3LjIzNjg0IDEyLjc2MzIgNi43NjMxNiAxMi40NzEgNi40NzEwMVY2LjQ3MTAxQzEyLjE3OSA2LjE3OTA1IDExLjcwNTcgNi4xNzg4NCAxMS40MTM1IDYuNDcwNTRMNy43NSAxMC4xMjc1VjEuNzVDNy43NSAxLjMzNTc5IDcuNDE0MjEgMSA3IDFWMUM2LjU4NTc5IDEgNi4yNSAxLjMzNTc5IDYuMjUgMS43NVYxMC4xMjc1TDIuNTk3MjYgNi40NjgyMkMyLjMwMzM4IDYuMTczODEgMS44MjY0MSA2LjE3MzU5IDEuNTMyMjYgNi40Njc3NFY2LjQ2Nzc0QzEuMjM4MyA2Ljc2MTcgMS4yMzgzIDcuMjM4MyAxLjUzMjI2IDcuNTMyMjZMNi4yOTI4OSAxMi4yOTI5QzYuNjgzNDIgMTIuNjgzNCA3LjMxNjU4IDEyLjY4MzQgNy43MDcxMSAxMi4yOTI5TDEyLjQ3MSA3LjUyODk5WiIgZmlsbD0iIzYxNjE2MSIvPgo8L3N2Zz4K);
  --jp-icon-move-up: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBkPSJNMS41Mjg5OSA2LjQ3MTAxQzEuMjM2ODQgNi43NjMxNiAxLjIzNjg0IDcuMjM2ODQgMS41Mjg5OSA3LjUyODk5VjcuNTI4OTlDMS44MjA5NSA3LjgyMDk1IDIuMjk0MjYgNy44MjExNiAyLjU4NjQ5IDcuNTI5NDZMNi4yNSAzLjg3MjVWMTIuMjVDNi4yNSAxMi42NjQyIDYuNTg1NzkgMTMgNyAxM1YxM0M3LjQxNDIxIDEzIDcuNzUgMTIuNjY0MiA3Ljc1IDEyLjI1VjMuODcyNUwxMS40MDI3IDcuNTMxNzhDMTEuNjk2NiA3LjgyNjE5IDEyLjE3MzYgNy44MjY0MSAxMi40Njc3IDcuNTMyMjZWNy41MzIyNkMxMi43NjE3IDcuMjM4MyAxMi43NjE3IDYuNzYxNyAxMi40Njc3IDYuNDY3NzRMNy43MDcxMSAxLjcwNzExQzcuMzE2NTggMS4zMTY1OCA2LjY4MzQyIDEuMzE2NTggNi4yOTI4OSAxLjcwNzExTDEuNTI4OTkgNi40NzEwMVoiIGZpbGw9IiM2MTYxNjEiLz4KPC9zdmc+Cg==);
  --jp-icon-new-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwIDZoLThsLTItMkg0Yy0xLjExIDAtMS45OS44OS0xLjk5IDJMMiAxOGMwIDEuMTEuODkgMiAyIDJoMTZjMS4xMSAwIDItLjg5IDItMlY4YzAtMS4xMS0uODktMi0yLTJ6bS0xIDhoLTN2M2gtMnYtM2gtM3YtMmgzVjloMnYzaDN2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-not-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI1IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMTkgMTcuMTg0NCAyLjk2OTY4IDE0LjMwMzIgMS44NjA5NCAxMS40NDA5WiIvPgogICAgPHBhdGggY2xhc3M9ImpwLWljb24yIiBzdHJva2U9IiMzMzMzMzMiIHN0cm9rZS13aWR0aD0iMiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOS4zMTU5MiA5LjMyMDMxKSIgZD0iTTcuMzY4NDIgMEwwIDcuMzY0NzkiLz4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDkuMzE1OTIgMTYuNjgzNikgc2NhbGUoMSAtMSkiIGQ9Ik03LjM2ODQyIDBMMCA3LjM2NDc5Ii8+Cjwvc3ZnPgo=);
  --jp-icon-notebook: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtbm90ZWJvb2staWNvbi1jb2xvciBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNFRjZDMDAiPgogICAgPHBhdGggZD0iTTE4LjcgMy4zdjE1LjRIMy4zVjMuM2gxNS40bTEuNS0xLjVIMS44djE4LjNoMTguM2wuMS0xOC4zeiIvPgogICAgPHBhdGggZD0iTTE2LjUgMTYuNWwtNS40LTQuMy01LjYgNC4zdi0xMWgxMXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-numbering: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIiIGhlaWdodD0iMjIiIHZpZXdCb3g9IjAgMCAyOCAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTQgMTlINlYxOS41SDVWMjAuNUg2VjIxSDRWMjJIN1YxOEg0VjE5Wk01IDEwSDZWNkg0VjdINVYxMFpNNCAxM0g1LjhMNCAxNS4xVjE2SDdWMTVINS4yTDcgMTIuOVYxMkg0VjEzWk05IDdWOUgyM1Y3SDlaTTkgMjFIMjNWMTlIOVYyMVpNOSAxNUgyM1YxM0g5VjE1WiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-offline-bolt: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyIDIuMDJjLTUuNTEgMC05Ljk4IDQuNDctOS45OCA5Ljk4czQuNDcgOS45OCA5Ljk4IDkuOTggOS45OC00LjQ3IDkuOTgtOS45OFMxNy41MSAyLjAyIDEyIDIuMDJ6TTExLjQ4IDIwdi02LjI2SDhMMTMgNHY2LjI2aDMuMzVMMTEuNDggMjB6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-palette: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE4IDEzVjIwSDRWNkg5LjAyQzkuMDcgNS4yOSA5LjI0IDQuNjIgOS41IDRINEMyLjkgNCAyIDQuOSAyIDZWMjBDMiAyMS4xIDIuOSAyMiA0IDIySDE4QzE5LjEgMjIgMjAgMjEuMSAyMCAyMFYxNUwxOCAxM1pNMTkuMyA4Ljg5QzE5Ljc0IDguMTkgMjAgNy4zOCAyMCA2LjVDMjAgNC4wMSAxNy45OSAyIDE1LjUgMkMxMy4wMSAyIDExIDQuMDEgMTEgNi41QzExIDguOTkgMTMuMDEgMTEgMTUuNDkgMTFDMTYuMzcgMTEgMTcuMTkgMTAuNzQgMTcuODggMTAuM0wyMSAxMy40MkwyMi40MiAxMkwxOS4zIDguODlaTTE1LjUgOUMxNC4xMiA5IDEzIDcuODggMTMgNi41QzEzIDUuMTIgMTQuMTIgNCAxNS41IDRDMTYuODggNCAxOCA1LjEyIDE4IDYuNUMxOCA3Ljg4IDE2Ljg4IDkgMTUuNSA5WiIvPgogICAgPHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik00IDZIOS4wMTg5NEM5LjAwNjM5IDYuMTY1MDIgOSA2LjMzMTc2IDkgNi41QzkgOC44MTU3NyAxMC4yMTEgMTAuODQ4NyAxMi4wMzQzIDEySDlWMTRIMTZWMTIuOTgxMUMxNi41NzAzIDEyLjkzNzcgMTcuMTIgMTIuODIwNyAxNy42Mzk2IDEyLjYzOTZMMTggMTNWMjBINFY2Wk04IDhINlYxMEg4VjhaTTYgMTJIOFYxNEg2VjEyWk04IDE2SDZWMThIOFYxNlpNOSAxNkgxNlYxOEg5VjE2WiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-paste: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE5IDJoLTQuMThDMTQuNC44NCAxMy4zIDAgMTIgMGMtMS4zIDAtMi40Ljg0LTIuODIgMkg1Yy0xLjEgMC0yIC45LTIgMnYxNmMwIDEuMS45IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjRjMC0xLjEtLjktMi0yLTJ6bS03IDBjLjU1IDAgMSAuNDUgMSAxcy0uNDUgMS0xIDEtMS0uNDUtMS0xIC40NS0xIDEtMXptNyAxOEg1VjRoMnYzaDEwVjRoMnYxNnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-pdf: url(data:image/svg+xml;base64,PHN2ZwogICB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyMiAyMiIgd2lkdGg9IjE2Ij4KICAgIDxwYXRoIHRyYW5zZm9ybT0icm90YXRlKDQ1KSIgY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI0ZGMkEyQSIKICAgICAgIGQ9Im0gMjIuMzQ0MzY5LC0zLjAxNjM2NDIgaCA1LjYzODYwNCB2IDEuNTc5MjQzMyBoIC0zLjU0OTIyNyB2IDEuNTA4NjkyOTkgaCAzLjMzNzU3NiBWIDEuNjUwODE1NCBoIC0zLjMzNzU3NiB2IDMuNDM1MjYxMyBoIC0yLjA4OTM3NyB6IG0gLTcuMTM2NDQ0LDEuNTc5MjQzMyB2IDQuOTQzOTU0MyBoIDAuNzQ4OTIgcSAxLjI4MDc2MSwwIDEuOTUzNzAzLC0wLjYzNDk1MzUgMC42NzgzNjksLTAuNjM0OTUzNSAwLjY3ODM2OSwtMS44NDUxNjQxIDAsLTEuMjA0NzgzNTUgLTAuNjcyOTQyLC0xLjgzNDMxMDExIC0wLjY3Mjk0MiwtMC42Mjk1MjY1OSAtMS45NTkxMywtMC42Mjk1MjY1OSB6IG0gLTIuMDg5Mzc3LC0xLjU3OTI0MzMgaCAyLjIwMzM0MyBxIDEuODQ1MTY0LDAgMi43NDYwMzksMC4yNjU5MjA3IDAuOTA2MzAxLDAuMjYwNDkzNyAxLjU1MjEwOCwwLjg5MDAyMDMgMC41Njk4MywwLjU0ODEyMjMgMC44NDY2MDUsMS4yNjQ0ODAwNiAwLjI3Njc3NCwwLjcxNjM1NzgxIDAuMjc2Nzc0LDEuNjIyNjU4OTQgMCwwLjkxNzE1NTEgLTAuMjc2Nzc0LDEuNjM4OTM5OSAtMC4yNzY3NzUsMC43MTYzNTc4IC0wLjg0NjYwNSwxLjI2NDQ4IC0wLjY1MTIzNCwwLjYyOTUyNjYgLTEuNTYyOTYyLDAuODk1NDQ3MyAtMC45MTE3MjgsMC4yNjA0OTM3IC0yLjczNTE4NSwwLjI2MDQ5MzcgaCAtMi4yMDMzNDMgeiBtIC04LjE0NTg1NjUsMCBoIDMuNDY3ODIzIHEgMS41NDY2ODE2LDAgMi4zNzE1Nzg1LDAuNjg5MjIzIDAuODMwMzI0LDAuNjgzNzk2MSAwLjgzMDMyNCwxLjk1MzcwMzE0IDAsMS4yNzUzMzM5NyAtMC44MzAzMjQsMS45NjQ1NTcwNiBRIDkuOTg3MTk2MSwyLjI3NDkxNSA4LjQ0MDUxNDUsMi4yNzQ5MTUgSCA3LjA2MjA2ODQgViA1LjA4NjA3NjcgSCA0Ljk3MjY5MTUgWiBtIDIuMDg5Mzc2OSwxLjUxNDExOTkgdiAyLjI2MzAzOTQzIGggMS4xNTU5NDEgcSAwLjYwNzgxODgsMCAwLjkzODg2MjksLTAuMjkzMDU1NDcgMC4zMzEwNDQxLC0wLjI5ODQ4MjQxIDAuMzMxMDQ0MSwtMC44NDExNzc3MiAwLC0wLjU0MjY5NTMxIC0wLjMzMTA0NDEsLTAuODM1NzUwNzQgLTAuMzMxMDQ0MSwtMC4yOTMwNTU1IC0wLjkzODg2MjksLTAuMjkzMDU1NSB6IgovPgo8L3N2Zz4K);
  --jp-icon-python: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iLTEwIC0xMCAxMzEuMTYxMzYxNjk0MzM1OTQgMTMyLjM4ODk5OTkzODk2NDg0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMzA2OTk4IiBkPSJNIDU0LjkxODc4NSw5LjE5Mjc0MjFlLTQgQyA1MC4zMzUxMzIsMC4wMjIyMTcyNyA0NS45NTc4NDYsMC40MTMxMzY5NyA0Mi4xMDYyODUsMS4wOTQ2NjkzIDMwLjc2MDA2OSwzLjA5OTE3MzEgMjguNzAwMDM2LDcuMjk0NzcxNCAyOC43MDAwMzUsMTUuMDMyMTY5IHYgMTAuMjE4NzUgaCAyNi44MTI1IHYgMy40MDYyNSBoIC0yNi44MTI1IC0xMC4wNjI1IGMgLTcuNzkyNDU5LDAgLTE0LjYxNTc1ODgsNC42ODM3MTcgLTE2Ljc0OTk5OTgsMTMuNTkzNzUgLTIuNDYxODE5OTgsMTAuMjEyOTY2IC0yLjU3MTAxNTA4LDE2LjU4NjAyMyAwLDI3LjI1IDEuOTA1OTI4Myw3LjkzNzg1MiA2LjQ1NzU0MzIsMTMuNTkzNzQ4IDE0LjI0OTk5OTgsMTMuNTkzNzUgaCA5LjIxODc1IHYgLTEyLjI1IGMgMCwtOC44NDk5MDIgNy42NTcxNDQsLTE2LjY1NjI0OCAxNi43NSwtMTYuNjU2MjUgaCAyNi43ODEyNSBjIDcuNDU0OTUxLDAgMTMuNDA2MjUzLC02LjEzODE2NCAxMy40MDYyNSwtMTMuNjI1IHYgLTI1LjUzMTI1IGMgMCwtNy4yNjYzMzg2IC02LjEyOTk4LC0xMi43MjQ3NzcxIC0xMy40MDYyNSwtMTMuOTM3NDk5NyBDIDY0LjI4MTU0OCwwLjMyNzk0Mzk3IDU5LjUwMjQzOCwtMC4wMjAzNzkwMyA1NC45MTg3ODUsOS4xOTI3NDIxZS00IFogbSAtMTQuNSw4LjIxODc1MDEyNTc5IGMgMi43Njk1NDcsMCA1LjAzMTI1LDIuMjk4NjQ1NiA1LjAzMTI1LDUuMTI0OTk5NiAtMmUtNiwyLjgxNjMzNiAtMi4yNjE3MDMsNS4wOTM3NSAtNS4wMzEyNSw1LjA5Mzc1IC0yLjc3OTQ3NiwtMWUtNiAtNS4wMzEyNSwtMi4yNzc0MTUgLTUuMDMxMjUsLTUuMDkzNzUgLTEwZS03LC0yLjgyNjM1MyAyLjI1MTc3NCwtNS4xMjQ5OTk2IDUuMDMxMjUsLTUuMTI0OTk5NiB6Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI2ZmZDQzYiIgZD0ibSA4NS42Mzc1MzUsMjguNjU3MTY5IHYgMTEuOTA2MjUgYyAwLDkuMjMwNzU1IC03LjgyNTg5NSwxNi45OTk5OTkgLTE2Ljc1LDE3IGggLTI2Ljc4MTI1IGMgLTcuMzM1ODMzLDAgLTEzLjQwNjI0OSw2LjI3ODQ4MyAtMTMuNDA2MjUsMTMuNjI1IHYgMjUuNTMxMjQ3IGMgMCw3LjI2NjM0NCA2LjMxODU4OCwxMS41NDAzMjQgMTMuNDA2MjUsMTMuNjI1MDA0IDguNDg3MzMxLDIuNDk1NjEgMTYuNjI2MjM3LDIuOTQ2NjMgMjYuNzgxMjUsMCA2Ljc1MDE1NSwtMS45NTQzOSAxMy40MDYyNTMsLTUuODg3NjEgMTMuNDA2MjUsLTEzLjYyNTAwNCBWIDg2LjUwMDkxOSBoIC0yNi43ODEyNSB2IC0zLjQwNjI1IGggMjYuNzgxMjUgMTMuNDA2MjU0IGMgNy43OTI0NjEsMCAxMC42OTYyNTEsLTUuNDM1NDA4IDEzLjQwNjI0MSwtMTMuNTkzNzUgMi43OTkzMywtOC4zOTg4ODYgMi42ODAyMiwtMTYuNDc1Nzc2IDAsLTI3LjI1IC0xLjkyNTc4LC03Ljc1NzQ0MSAtNS42MDM4NywtMTMuNTkzNzUgLTEzLjQwNjI0MSwtMTMuNTkzNzUgeiBtIC0xNS4wNjI1LDY0LjY1NjI1IGMgMi43Nzk0NzgsM2UtNiA1LjAzMTI1LDIuMjc3NDE3IDUuMDMxMjUsNS4wOTM3NDcgLTJlLTYsMi44MjYzNTQgLTIuMjUxNzc1LDUuMTI1MDA0IC01LjAzMTI1LDUuMTI1MDA0IC0yLjc2OTU1LDAgLTUuMDMxMjUsLTIuMjk4NjUgLTUuMDMxMjUsLTUuMTI1MDA0IDJlLTYsLTIuODE2MzMgMi4yNjE2OTcsLTUuMDkzNzQ3IDUuMDMxMjUsLTUuMDkzNzQ3IHoiLz4KPC9zdmc+Cg==);
  --jp-icon-r-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjE5NkYzIiBkPSJNNC40IDIuNWMxLjItLjEgMi45LS4zIDQuOS0uMyAyLjUgMCA0LjEuNCA1LjIgMS4zIDEgLjcgMS41IDEuOSAxLjUgMy41IDAgMi0xLjQgMy41LTIuOSA0LjEgMS4yLjQgMS43IDEuNiAyLjIgMyAuNiAxLjkgMSAzLjkgMS4zIDQuNmgtMy44Yy0uMy0uNC0uOC0xLjctMS4yLTMuN3MtMS4yLTIuNi0yLjYtMi42aC0uOXY2LjRINC40VjIuNXptMy43IDYuOWgxLjRjMS45IDAgMi45LS45IDIuOS0yLjNzLTEtMi4zLTIuOC0yLjNjLS43IDAtMS4zIDAtMS42LjJ2NC41aC4xdi0uMXoiLz4KPC9zdmc+Cg==);
  --jp-icon-react: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMTUwIDE1MCA1NDEuOSAyOTUuMyI+CiAgPGcgY2xhc3M9ImpwLWljb24tYnJhbmQyIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzYxREFGQiI+CiAgICA8cGF0aCBkPSJNNjY2LjMgMjk2LjVjMC0zMi41LTQwLjctNjMuMy0xMDMuMS04Mi40IDE0LjQtNjMuNiA4LTExNC4yLTIwLjItMTMwLjQtNi41LTMuOC0xNC4xLTUuNi0yMi40LTUuNnYyMi4zYzQuNiAwIDguMy45IDExLjQgMi42IDEzLjYgNy44IDE5LjUgMzcuNSAxNC45IDc1LjctMS4xIDkuNC0yLjkgMTkuMy01LjEgMjkuNC0xOS42LTQuOC00MS04LjUtNjMuNS0xMC45LTEzLjUtMTguNS0yNy41LTM1LjMtNDEuNi01MCAzMi42LTMwLjMgNjMuMi00Ni45IDg0LTQ2LjlWNzhjLTI3LjUgMC02My41IDE5LjYtOTkuOSA1My42LTM2LjQtMzMuOC03Mi40LTUzLjItOTkuOS01My4ydjIyLjNjMjAuNyAwIDUxLjQgMTYuNSA4NCA0Ni42LTE0IDE0LjctMjggMzEuNC00MS4zIDQ5LjktMjIuNiAyLjQtNDQgNi4xLTYzLjYgMTEtMi4zLTEwLTQtMTkuNy01LjItMjktNC43LTM4LjIgMS4xLTY3LjkgMTQuNi03NS44IDMtMS44IDYuOS0yLjYgMTEuNS0yLjZWNzguNWMtOC40IDAtMTYgMS44LTIyLjYgNS42LTI4LjEgMTYuMi0zNC40IDY2LjctMTkuOSAxMzAuMS02Mi4yIDE5LjItMTAyLjcgNDkuOS0xMDIuNyA4Mi4zIDAgMzIuNSA0MC43IDYzLjMgMTAzLjEgODIuNC0xNC40IDYzLjYtOCAxMTQuMiAyMC4yIDEzMC40IDYuNSAzLjggMTQuMSA1LjYgMjIuNSA1LjYgMjcuNSAwIDYzLjUtMTkuNiA5OS45LTUzLjYgMzYuNCAzMy44IDcyLjQgNTMuMiA5OS45IDUzLjIgOC40IDAgMTYtMS44IDIyLjYtNS42IDI4LjEtMTYuMiAzNC40LTY2LjcgMTkuOS0xMzAuMSA2Mi0xOS4xIDEwMi41LTQ5LjkgMTAyLjUtODIuM3ptLTEzMC4yLTY2LjdjLTMuNyAxMi45LTguMyAyNi4yLTEzLjUgMzkuNS00LjEtOC04LjQtMTYtMTMuMS0yNC00LjYtOC05LjUtMTUuOC0xNC40LTIzLjQgMTQuMiAyLjEgMjcuOSA0LjcgNDEgNy45em0tNDUuOCAxMDYuNWMtNy44IDEzLjUtMTUuOCAyNi4zLTI0LjEgMzguMi0xNC45IDEuMy0zMCAyLTQ1LjIgMi0xNS4xIDAtMzAuMi0uNy00NS0xLjktOC4zLTExLjktMTYuNC0yNC42LTI0LjItMzgtNy42LTEzLjEtMTQuNS0yNi40LTIwLjgtMzkuOCA2LjItMTMuNCAxMy4yLTI2LjggMjAuNy0zOS45IDcuOC0xMy41IDE1LjgtMjYuMyAyNC4xLTM4LjIgMTQuOS0xLjMgMzAtMiA0NS4yLTIgMTUuMSAwIDMwLjIuNyA0NSAxLjkgOC4zIDExLjkgMTYuNCAyNC42IDI0LjIgMzggNy42IDEzLjEgMTQuNSAyNi40IDIwLjggMzkuOC02LjMgMTMuNC0xMy4yIDI2LjgtMjAuNyAzOS45em0zMi4zLTEzYzUuNCAxMy40IDEwIDI2LjggMTMuOCAzOS44LTEzLjEgMy4yLTI2LjkgNS45LTQxLjIgOCA0LjktNy43IDkuOC0xNS42IDE0LjQtMjMuNyA0LjYtOCA4LjktMTYuMSAxMy0yNC4xek00MjEuMiA0MzBjLTkuMy05LjYtMTguNi0yMC4zLTI3LjgtMzIgOSAuNCAxOC4yLjcgMjcuNS43IDkuNCAwIDE4LjctLjIgMjcuOC0uNy05IDExLjctMTguMyAyMi40LTI3LjUgMzJ6bS03NC40LTU4LjljLTE0LjItMi4xLTI3LjktNC43LTQxLTcuOSAzLjctMTIuOSA4LjMtMjYuMiAxMy41LTM5LjUgNC4xIDggOC40IDE2IDEzLjEgMjQgNC43IDggOS41IDE1LjggMTQuNCAyMy40ek00MjAuNyAxNjNjOS4zIDkuNiAxOC42IDIwLjMgMjcuOCAzMi05LS40LTE4LjItLjctMjcuNS0uNy05LjQgMC0xOC43LjItMjcuOC43IDktMTEuNyAxOC4zLTIyLjQgMjcuNS0zMnptLTc0IDU4LjljLTQuOSA3LjctOS44IDE1LjYtMTQuNCAyMy43LTQuNiA4LTguOSAxNi0xMyAyNC01LjQtMTMuNC0xMC0yNi44LTEzLjgtMzkuOCAxMy4xLTMuMSAyNi45LTUuOCA0MS4yLTcuOXptLTkwLjUgMTI1LjJjLTM1LjQtMTUuMS01OC4zLTM0LjktNTguMy01MC42IDAtMTUuNyAyMi45LTM1LjYgNTguMy01MC42IDguNi0zLjcgMTgtNyAyNy43LTEwLjEgNS43IDE5LjYgMTMuMiA0MCAyMi41IDYwLjktOS4yIDIwLjgtMTYuNiA0MS4xLTIyLjIgNjAuNi05LjktMy4xLTE5LjMtNi41LTI4LTEwLjJ6TTMxMCA0OTBjLTEzLjYtNy44LTE5LjUtMzcuNS0xNC45LTc1LjcgMS4xLTkuNCAyLjktMTkuMyA1LjEtMjkuNCAxOS42IDQuOCA0MSA4LjUgNjMuNSAxMC45IDEzLjUgMTguNSAyNy41IDM1LjMgNDEuNiA1MC0zMi42IDMwLjMtNjMuMiA0Ni45LTg0IDQ2LjktNC41LS4xLTguMy0xLTExLjMtMi43em0yMzcuMi03Ni4yYzQuNyAzOC4yLTEuMSA2Ny45LTE0LjYgNzUuOC0zIDEuOC02LjkgMi42LTExLjUgMi42LTIwLjcgMC01MS40LTE2LjUtODQtNDYuNiAxNC0xNC43IDI4LTMxLjQgNDEuMy00OS45IDIyLjYtMi40IDQ0LTYuMSA2My42LTExIDIuMyAxMC4xIDQuMSAxOS44IDUuMiAyOS4xem0zOC41LTY2LjdjLTguNiAzLjctMTggNy0yNy43IDEwLjEtNS43LTE5LjYtMTMuMi00MC0yMi41LTYwLjkgOS4yLTIwLjggMTYuNi00MS4xIDIyLjItNjAuNiA5LjkgMy4xIDE5LjMgNi41IDI4LjEgMTAuMiAzNS40IDE1LjEgNTguMyAzNC45IDU4LjMgNTAuNi0uMSAxNS43LTIzIDM1LjYtNTguNCA1MC42ek0zMjAuOCA3OC40eiIvPgogICAgPGNpcmNsZSBjeD0iNDIwLjkiIGN5PSIyOTYuNSIgcj0iNDUuNyIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-redo: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgICA8cGF0aCBkPSJNMCAwaDI0djI0SDB6IiBmaWxsPSJub25lIi8+PHBhdGggZD0iTTE4LjQgMTAuNkMxNi41NSA4Ljk5IDE0LjE1IDggMTEuNSA4Yy00LjY1IDAtOC41OCAzLjAzLTkuOTYgNy4yMkwzLjkgMTZjMS4wNS0zLjE5IDQuMDUtNS41IDcuNi01LjUgMS45NSAwIDMuNzMuNzIgNS4xMiAxLjg4TDEzIDE2aDlWN2wtMy42IDMuNnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-refresh: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTkgMTMuNWMtMi40OSAwLTQuNS0yLjAxLTQuNS00LjVTNi41MSA0LjUgOSA0LjVjMS4yNCAwIDIuMzYuNTIgMy4xNyAxLjMzTDEwIDhoNVYzbC0xLjc2IDEuNzZDMTIuMTUgMy42OCAxMC42NiAzIDkgMyA1LjY5IDMgMy4wMSA1LjY5IDMuMDEgOVM1LjY5IDE1IDkgMTVjMi45NyAwIDUuNDMtMi4xNiA1LjktNWgtMS41MmMtLjQ2IDItMi4yNCAzLjUtNC4zOCAzLjV6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-regex: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi1hY2NlbnQyIiBmaWxsPSIjRkZGIj4KICAgIDxjaXJjbGUgY2xhc3M9InN0MiIgY3g9IjUuNSIgY3k9IjE0LjUiIHI9IjEuNSIvPgogICAgPHJlY3QgeD0iMTIiIHk9IjQiIGNsYXNzPSJzdDIiIHdpZHRoPSIxIiBoZWlnaHQ9IjgiLz4KICAgIDxyZWN0IHg9IjguNSIgeT0iNy41IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjg2NiAtMC41IDAuNSAwLjg2NiAtMi4zMjU1IDcuMzIxOSkiIGNsYXNzPSJzdDIiIHdpZHRoPSI4IiBoZWlnaHQ9IjEiLz4KICAgIDxyZWN0IHg9IjEyIiB5PSI0IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjUgLTAuODY2IDAuODY2IDAuNSAtMC42Nzc5IDE0LjgyNTIpIiBjbGFzcz0ic3QyIiB3aWR0aD0iMSIgaGVpZ2h0PSI4Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-run: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTggNXYxNGwxMS03eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-running: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMjU2IDhDMTE5IDggOCAxMTkgOCAyNTZzMTExIDI0OCAyNDggMjQ4IDI0OC0xMTEgMjQ4LTI0OFMzOTMgOCAyNTYgOHptOTYgMzI4YzAgOC44LTcuMiAxNi0xNiAxNkgxNzZjLTguOCAwLTE2LTcuMi0xNi0xNlYxNzZjMC04LjggNy4yLTE2IDE2LTE2aDE2MGM4LjggMCAxNiA3LjIgMTYgMTZ2MTYweiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-save: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE3IDNINWMtMS4xMSAwLTIgLjktMiAydjE0YzAgMS4xLjg5IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjdsLTQtNHptLTUgMTZjLTEuNjYgMC0zLTEuMzQtMy0zczEuMzQtMyAzLTMgMyAxLjM0IDMgMy0xLjM0IDMtMyAzem0zLTEwSDVWNWgxMHY0eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-search: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjEsMTAuOWgtMC43bC0wLjItMC4yYzAuOC0wLjksMS4zLTIuMiwxLjMtMy41YzAtMy0yLjQtNS40LTUuNC01LjRTMS44LDQuMiwxLjgsNy4xczIuNCw1LjQsNS40LDUuNCBjMS4zLDAsMi41LTAuNSwzLjUtMS4zbDAuMiwwLjJ2MC43bDQuMSw0LjFsMS4yLTEuMkwxMi4xLDEwLjl6IE03LjEsMTAuOWMtMi4xLDAtMy43LTEuNy0zLjctMy43czEuNy0zLjcsMy43LTMuN3MzLjcsMS43LDMuNywzLjcgUzkuMiwxMC45LDcuMSwxMC45eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-settings: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuNDMgMTIuOThjLjA0LS4zMi4wNy0uNjQuMDctLjk4cy0uMDMtLjY2LS4wNy0uOThsMi4xMS0xLjY1Yy4xOS0uMTUuMjQtLjQyLjEyLS42NGwtMi0zLjQ2Yy0uMTItLjIyLS4zOS0uMy0uNjEtLjIybC0yLjQ5IDFjLS41Mi0uNC0xLjA4LS43My0xLjY5LS45OGwtLjM4LTIuNjVBLjQ4OC40ODggMCAwMDE0IDJoLTRjLS4yNSAwLS40Ni4xOC0uNDkuNDJsLS4zOCAyLjY1Yy0uNjEuMjUtMS4xNy41OS0xLjY5Ljk4bC0yLjQ5LTFjLS4yMy0uMDktLjQ5IDAtLjYxLjIybC0yIDMuNDZjLS4xMy4yMi0uMDcuNDkuMTIuNjRsMi4xMSAxLjY1Yy0uMDQuMzItLjA3LjY1LS4wNy45OHMuMDMuNjYuMDcuOThsLTIuMTEgMS42NWMtLjE5LjE1LS4yNC40Mi0uMTIuNjRsMiAzLjQ2Yy4xMi4yMi4zOS4zLjYxLjIybDIuNDktMWMuNTIuNCAxLjA4LjczIDEuNjkuOThsLjM4IDIuNjVjLjAzLjI0LjI0LjQyLjQ5LjQyaDRjLjI1IDAgLjQ2LS4xOC40OS0uNDJsLjM4LTIuNjVjLjYxLS4yNSAxLjE3LS41OSAxLjY5LS45OGwyLjQ5IDFjLjIzLjA5LjQ5IDAgLjYxLS4yMmwyLTMuNDZjLjEyLS4yMi4wNy0uNDktLjEyLS42NGwtMi4xMS0xLjY1ek0xMiAxNS41Yy0xLjkzIDAtMy41LTEuNTctMy41LTMuNXMxLjU3LTMuNSAzLjUtMy41IDMuNSAxLjU3IDMuNSAzLjUtMS41NyAzLjUtMy41IDMuNXoiLz4KPC9zdmc+Cg==);
  --jp-icon-share: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTSAxOCAyIEMgMTYuMzU0OTkgMiAxNSAzLjM1NDk5MDQgMTUgNSBDIDE1IDUuMTkwOTUyOSAxNS4wMjE3OTEgNS4zNzcxMjI0IDE1LjA1NjY0MSA1LjU1ODU5MzggTCA3LjkyMTg3NSA5LjcyMDcwMzEgQyA3LjM5ODUzOTkgOS4yNzc4NTM5IDYuNzMyMDc3MSA5IDYgOSBDIDQuMzU0OTkwNCA5IDMgMTAuMzU0OTkgMyAxMiBDIDMgMTMuNjQ1MDEgNC4zNTQ5OTA0IDE1IDYgMTUgQyA2LjczMjA3NzEgMTUgNy4zOTg1Mzk5IDE0LjcyMjE0NiA3LjkyMTg3NSAxNC4yNzkyOTcgTCAxNS4wNTY2NDEgMTguNDM5NDUzIEMgMTUuMDIxNTU1IDE4LjYyMTUxNCAxNSAxOC44MDgzODYgMTUgMTkgQyAxNSAyMC42NDUwMSAxNi4zNTQ5OSAyMiAxOCAyMiBDIDE5LjY0NTAxIDIyIDIxIDIwLjY0NTAxIDIxIDE5IEMgMjEgMTcuMzU0OTkgMTkuNjQ1MDEgMTYgMTggMTYgQyAxNy4yNjc0OCAxNiAxNi42MDE1OTMgMTYuMjc5MzI4IDE2LjA3ODEyNSAxNi43MjI2NTYgTCA4Ljk0MzM1OTQgMTIuNTU4NTk0IEMgOC45NzgyMDk1IDEyLjM3NzEyMiA5IDEyLjE5MDk1MyA5IDEyIEMgOSAxMS44MDkwNDcgOC45NzgyMDk1IDExLjYyMjg3OCA4Ljk0MzM1OTQgMTEuNDQxNDA2IEwgMTYuMDc4MTI1IDcuMjc5Mjk2OSBDIDE2LjYwMTQ2IDcuNzIyMTQ2MSAxNy4yNjc5MjMgOCAxOCA4IEMgMTkuNjQ1MDEgOCAyMSA2LjY0NTAwOTYgMjEgNSBDIDIxIDMuMzU0OTkwNCAxOS42NDUwMSAyIDE4IDIgeiBNIDE4IDQgQyAxOC41NjQxMjkgNCAxOSA0LjQzNTg3MDYgMTkgNSBDIDE5IDUuNTY0MTI5NCAxOC41NjQxMjkgNiAxOCA2IEMgMTcuNDM1ODcxIDYgMTcgNS41NjQxMjk0IDE3IDUgQyAxNyA0LjQzNTg3MDYgMTcuNDM1ODcxIDQgMTggNCB6IE0gNiAxMSBDIDYuNTY0MTI5NCAxMSA3IDExLjQzNTg3MSA3IDEyIEMgNyAxMi41NjQxMjkgNi41NjQxMjk0IDEzIDYgMTMgQyA1LjQzNTg3MDYgMTMgNSAxMi41NjQxMjkgNSAxMiBDIDUgMTEuNDM1ODcxIDUuNDM1ODcwNiAxMSA2IDExIHogTSAxOCAxOCBDIDE4LjU2NDEyOSAxOCAxOSAxOC40MzU4NzEgMTkgMTkgQyAxOSAxOS41NjQxMjkgMTguNTY0MTI5IDIwIDE4IDIwIEMgMTcuNDM1ODcxIDIwIDE3IDE5LjU2NDEyOSAxNyAxOSBDIDE3IDE4LjQzNTg3MSAxNy40MzU4NzEgMTggMTggMTggeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-spreadsheet: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNENBRjUwIiBkPSJNMi4yIDIuMnYxNy42aDE3LjZWMi4ySDIuMnptMTUuNCA3LjdoLTUuNVY0LjRoNS41djUuNXpNOS45IDQuNHY1LjVINC40VjQuNGg1LjV6bS01LjUgNy43aDUuNXY1LjVINC40di01LjV6bTcuNyA1LjV2LTUuNWg1LjV2NS41aC01LjV6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-stop: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik02IDZoMTJ2MTJINnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-tab: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIxIDNIM2MtMS4xIDAtMiAuOS0yIDJ2MTRjMCAxLjEuOSAyIDIgMmgxOGMxLjEgMCAyLS45IDItMlY1YzAtMS4xLS45LTItMi0yem0wIDE2SDNWNWgxMHY0aDh2MTB6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-table-rows: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik0yMSw4SDNWNGgxOFY4eiBNMjEsMTBIM3Y0aDE4VjEweiBNMjEsMTZIM3Y0aDE4VjE2eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-tag: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjgiIGhlaWdodD0iMjgiIHZpZXdCb3g9IjAgMCA0MyAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTI4LjgzMzIgMTIuMzM0TDMyLjk5OTggMTYuNTAwN0wzNy4xNjY1IDEyLjMzNEgyOC44MzMyWiIvPgoJCTxwYXRoIGQ9Ik0xNi4yMDk1IDIxLjYxMDRDMTUuNjg3MyAyMi4xMjk5IDE0Ljg0NDMgMjIuMTI5OSAxNC4zMjQ4IDIxLjYxMDRMNi45ODI5IDE0LjcyNDVDNi41NzI0IDE0LjMzOTQgNi4wODMxMyAxMy42MDk4IDYuMDQ3ODYgMTMuMDQ4MkM1Ljk1MzQ3IDExLjUyODggNi4wMjAwMiA4LjYxOTQ0IDYuMDY2MjEgNy4wNzY5NUM2LjA4MjgxIDYuNTE0NzcgNi41NTU0OCA2LjA0MzQ3IDcuMTE4MDQgNi4wMzA1NUM5LjA4ODYzIDUuOTg0NzMgMTMuMjYzOCA1LjkzNTc5IDEzLjY1MTggNi4zMjQyNUwyMS43MzY5IDEzLjYzOUMyMi4yNTYgMTQuMTU4NSAyMS43ODUxIDE1LjQ3MjQgMjEuMjYyIDE1Ljk5NDZMMTYuMjA5NSAyMS42MTA0Wk05Ljc3NTg1IDguMjY1QzkuMzM1NTEgNy44MjU2NiA4LjYyMzUxIDcuODI1NjYgOC4xODI4IDguMjY1QzcuNzQzNDYgOC43MDU3MSA3Ljc0MzQ2IDkuNDE3MzMgOC4xODI4IDkuODU2NjdDOC42MjM4MiAxMC4yOTY0IDkuMzM1ODIgMTAuMjk2NCA5Ljc3NTg1IDkuODU2NjdDMTAuMjE1NiA5LjQxNzMzIDEwLjIxNTYgOC43MDUzMyA5Ljc3NTg1IDguMjY1WiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-terminal: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiA+CiAgICA8cmVjdCBjbGFzcz0ianAtdGVybWluYWwtaWNvbi1iYWNrZ3JvdW5kLWNvbG9yIGpwLWljb24tc2VsZWN0YWJsZSIgd2lkdGg9IjIwIiBoZWlnaHQ9IjIwIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgyIDIpIiBmaWxsPSIjMzMzMzMzIi8+CiAgICA8cGF0aCBjbGFzcz0ianAtdGVybWluYWwtaWNvbi1jb2xvciBqcC1pY29uLXNlbGVjdGFibGUtaW52ZXJzZSIgZD0iTTUuMDU2NjQgOC43NjE3MkM1LjA1NjY0IDguNTk3NjYgNS4wMzEyNSA4LjQ1MzEyIDQuOTgwNDcgOC4zMjgxMkM0LjkzMzU5IDguMTk5MjIgNC44NTU0NyA4LjA4MjAzIDQuNzQ2MDkgNy45NzY1NkM0LjY0MDYyIDcuODcxMDkgNC41IDcuNzc1MzkgNC4zMjQyMiA3LjY4OTQ1QzQuMTUyMzQgNy41OTk2MSAzLjk0MzM2IDcuNTExNzIgMy42OTcyNyA3LjQyNTc4QzMuMzAyNzMgNy4yODUxNiAyLjk0MzM2IDcuMTM2NzIgMi42MTkxNCA2Ljk4MDQ3QzIuMjk0OTIgNi44MjQyMiAyLjAxNzU4IDYuNjQyNTggMS43ODcxMSA2LjQzNTU1QzEuNTYwNTUgNi4yMjg1MiAxLjM4NDc3IDUuOTg4MjggMS4yNTk3NyA1LjcxNDg0QzEuMTM0NzcgNS40Mzc1IDEuMDcyMjcgNS4xMDkzOCAxLjA3MjI3IDQuNzMwNDdDMS4wNzIyNyA0LjM5ODQ0IDEuMTI4OTEgNC4wOTU3IDEuMjQyMTkgMy44MjIyN0MxLjM1NTQ3IDMuNTQ0OTIgMS41MTU2MiAzLjMwNDY5IDEuNzIyNjYgMy4xMDE1NkMxLjkyOTY5IDIuODk4NDQgMi4xNzk2OSAyLjczNDM3IDIuNDcyNjYgMi42MDkzOEMyLjc2NTYyIDIuNDg0MzggMy4wOTE4IDIuNDA0MyAzLjQ1MTE3IDIuMzY5MTRWMS4xMDkzOEg0LjM4ODY3VjIuMzgwODZDNC43NDAyMyAyLjQyNzczIDUuMDU2NjQgMi41MjM0NCA1LjMzNzg5IDIuNjY3OTdDNS42MTkxNCAyLjgxMjUgNS44NTc0MiAzLjAwMTk1IDYuMDUyNzMgMy4yMzYzM0M2LjI1MTk1IDMuNDY2OCA2LjQwNDMgMy43NDAyMyA2LjUwOTc3IDQuMDU2NjRDNi42MTkxNCA0LjM2OTE0IDYuNjczODMgNC43MjA3IDYuNjczODMgNS4xMTEzM0g1LjA0NDkyQzUuMDQ0OTIgNC42Mzg2NyA0LjkzNzUgNC4yODEyNSA0LjcyMjY2IDQuMDM5MDZDNC41MDc4MSAzLjc5Mjk3IDQuMjE2OCAzLjY2OTkyIDMuODQ5NjEgMy42Njk5MkMzLjY1MDM5IDMuNjY5OTIgMy40NzY1NiAzLjY5NzI3IDMuMzI4MTIgMy43NTE5NUMzLjE4MzU5IDMuODAyNzMgMy4wNjQ0NSAzLjg3Njk1IDIuOTcwNyAzLjk3NDYxQzIuODc2OTUgNC4wNjgzNiAyLjgwNjY0IDQuMTc5NjkgMi43NTk3NyA0LjMwODU5QzIuNzE2OCA0LjQzNzUgMi42OTUzMSA0LjU3ODEyIDIuNjk1MzEgNC43MzA0N0MyLjY5NTMxIDQuODgyODEgMi43MTY4IDUuMDE5NTMgMi43NTk3NyA1LjE0MDYyQzIuODA2NjQgNS4yNTc4MSAyLjg4MjgxIDUuMzY3MTkgMi45ODgyOCA1LjQ2ODc1QzMuMDk3NjYgNS41NzAzMSAzLjI0MDIzIDUuNjY3OTcgMy40MTYwMiA1Ljc2MTcyQzMuNTkxOCA1Ljg1MTU2IDMuODEwNTUgNS45NDMzNiA0LjA3MjI3IDYuMDM3MTFDNC40NjY4IDYuMTg1NTUgNC44MjQyMiA2LjMzOTg0IDUuMTQ0NTMgNi41QzUuNDY0ODQgNi42NTYyNSA1LjczODI4IDYuODM5ODQgNS45NjQ4NCA3LjA1MDc4QzYuMTk1MzEgNy4yNTc4MSA2LjM3MTA5IDcuNSA2LjQ5MjE5IDcuNzc3MzRDNi42MTcxOSA4LjA1MDc4IDYuNjc5NjkgOC4zNzUgNi42Nzk2OSA4Ljc1QzYuNjc5NjkgOS4wOTM3NSA2LjYyMzA1IDkuNDA0MyA2LjUwOTc3IDkuNjgxNjRDNi4zOTY0OCA5Ljk1NTA4IDYuMjM0MzggMTAuMTkxNCA2LjAyMzQ0IDEwLjM5MDZDNS44MTI1IDEwLjU4OTggNS41NTg1OSAxMC43NSA1LjI2MTcyIDEwLjg3MTFDNC45NjQ4NCAxMC45ODgzIDQuNjMyODEgMTEuMDY0NSA0LjI2NTYyIDExLjA5OTZWMTIuMjQ4SDMuMzMzOThWMTEuMDk5NkMzLjAwMTk1IDExLjA2ODQgMi42Nzk2OSAxMC45OTYxIDIuMzY3MTkgMTAuODgyOEMyLjA1NDY5IDEwLjc2NTYgMS43NzczNCAxMC41OTc3IDEuNTM1MTYgMTAuMzc4OUMxLjI5Njg4IDEwLjE2MDIgMS4xMDU0NyA5Ljg4NDc3IDAuOTYwOTM4IDkuNTUyNzNDMC44MTY0MDYgOS4yMTY4IDAuNzQ0MTQxIDguODE0NDUgMC43NDQxNDEgOC4zNDU3SDIuMzc4OTFDMi4zNzg5MSA4LjYyNjk1IDIuNDE5OTIgOC44NjMyOCAyLjUwMTk1IDkuMDU0NjlDMi41ODM5OCA5LjI0MjE5IDIuNjg5NDUgOS4zOTI1OCAyLjgxODM2IDkuNTA1ODZDMi45NTExNyA5LjYxNTIzIDMuMTAxNTYgOS42OTMzNiAzLjI2OTUzIDkuNzQwMjNDMy40Mzc1IDkuNzg3MTEgMy42MDkzOCA5LjgxMDU1IDMuNzg1MTYgOS44MTA1NUM0LjIwMzEyIDkuODEwNTUgNC41MTk1MyA5LjcxMjg5IDQuNzM0MzggOS41MTc1OEM0Ljk0OTIyIDkuMzIyMjcgNS4wNTY2NCA5LjA3MDMxIDUuMDU2NjQgOC43NjE3MlpNMTMuNDE4IDEyLjI3MTVIOC4wNzQyMlYxMUgxMy40MThWMTIuMjcxNVoiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMuOTUyNjQgNikiIGZpbGw9IndoaXRlIi8+Cjwvc3ZnPgo=);
  --jp-icon-text-editor: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtdGV4dC1lZGl0b3ItaWNvbi1jb2xvciBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xNSAxNUgzdjJoMTJ2LTJ6bTAtOEgzdjJoMTJWN3pNMyAxM2gxOHYtMkgzdjJ6bTAgOGgxOHYtMkgzdjJ6TTMgM3YyaDE4VjNIM3oiLz4KPC9zdmc+Cg==);
  --jp-icon-toc: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik03LDVIMjFWN0g3VjVNNywxM1YxMUgyMVYxM0g3TTQsNC41QTEuNSwxLjUgMCAwLDEgNS41LDZBMS41LDEuNSAwIDAsMSA0LDcuNUExLjUsMS41IDAgMCwxIDIuNSw2QTEuNSwxLjUgMCAwLDEgNCw0LjVNNCwxMC41QTEuNSwxLjUgMCAwLDEgNS41LDEyQTEuNSwxLjUgMCAwLDEgNCwxMy41QTEuNSwxLjUgMCAwLDEgMi41LDEyQTEuNSwxLjUgMCAwLDEgNCwxMC41TTcsMTlWMTdIMjFWMTlIN000LDE2LjVBMS41LDEuNSAwIDAsMSA1LjUsMThBMS41LDEuNSAwIDAsMSA0LDE5LjVBMS41LDEuNSAwIDAsMSAyLjUsMThBMS41LDEuNSAwIDAsMSA0LDE2LjVaIiAvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-tree-view: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik0yMiAxMVYzaC03djNIOVYzSDJ2OGg3VjhoMnYxMGg0djNoN3YtOGgtN3YzaC0yVjhoMnYzeiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMiAxNy4xODQ0IDIuOTY5NjggMTQuMzAzMiAxLjg2MDk0IDExLjQ0MDlaIi8+CiAgICA8cGF0aCBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiMzMzMzMzMiIHN0cm9rZT0iIzMzMzMzMyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOCA5Ljg2NzE5KSIgZD0iTTIuODYwMTUgNC44NjUzNUwwLjcyNjU0OSAyLjk5OTU5TDAgMy42MzA0NUwyLjg2MDE1IDYuMTMxNTdMOCAwLjYzMDg3Mkw3LjI3ODU3IDBMMi44NjAxNSA0Ljg2NTM1WiIvPgo8L3N2Zz4K);
  --jp-icon-undo: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjUgOGMtMi42NSAwLTUuMDUuOTktNi45IDIuNkwyIDd2OWg5bC0zLjYyLTMuNjJjMS4zOS0xLjE2IDMuMTYtMS44OCA1LjEyLTEuODggMy41NCAwIDYuNTUgMi4zMSA3LjYgNS41bDIuMzctLjc4QzIxLjA4IDExLjAzIDE3LjE1IDggMTIuNSA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-user: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE2IDdhNCA0IDAgMTEtOCAwIDQgNCAwIDAxOCAwek0xMiAxNGE3IDcgMCAwMC03IDdoMTRhNyA3IDAgMDAtNy03eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-users: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZlcnNpb249IjEuMSIgdmlld0JveD0iMCAwIDM2IDI0IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogPGcgY2xhc3M9ImpwLWljb24zIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjczMjcgMCAwIDEuNzMyNyAtMy42MjgyIC4wOTk1NzcpIiBmaWxsPSIjNjE2MTYxIj4KICA8cGF0aCB0cmFuc2Zvcm09Im1hdHJpeCgxLjUsMCwwLDEuNSwwLC02KSIgZD0ibTEyLjE4NiA3LjUwOThjLTEuMDUzNSAwLTEuOTc1NyAwLjU2NjUtMi40Nzg1IDEuNDEwMiAwLjc1MDYxIDAuMzEyNzcgMS4zOTc0IDAuODI2NDggMS44NzMgMS40NzI3aDMuNDg2M2MwLTEuNTkyLTEuMjg4OS0yLjg4MjgtMi44ODA5LTIuODgyOHoiLz4KICA8cGF0aCBkPSJtMjAuNDY1IDIuMzg5NWEyLjE4ODUgMi4xODg1IDAgMCAxLTIuMTg4NCAyLjE4ODUgMi4xODg1IDIuMTg4NSAwIDAgMS0yLjE4ODUtMi4xODg1IDIuMTg4NSAyLjE4ODUgMCAwIDEgMi4xODg1LTIuMTg4NSAyLjE4ODUgMi4xODg1IDAgMCAxIDIuMTg4NCAyLjE4ODV6Ii8+CiAgPHBhdGggdHJhbnNmb3JtPSJtYXRyaXgoMS41LDAsMCwxLjUsMCwtNikiIGQ9Im0zLjU4OTggOC40MjE5Yy0xLjExMjYgMC0yLjAxMzcgMC45MDExMS0yLjAxMzcgMi4wMTM3aDIuODE0NWMwLjI2Nzk3LTAuMzczMDkgMC41OTA3LTAuNzA0MzUgMC45NTg5OC0wLjk3ODUyLTAuMzQ0MzMtMC42MTY4OC0xLjAwMzEtMS4wMzUyLTEuNzU5OC0xLjAzNTJ6Ii8+CiAgPHBhdGggZD0ibTYuOTE1NCA0LjYyM2ExLjUyOTQgMS41Mjk0IDAgMCAxLTEuNTI5NCAxLjUyOTQgMS41Mjk0IDEuNTI5NCAwIDAgMS0xLjUyOTQtMS41Mjk0IDEuNTI5NCAxLjUyOTQgMCAwIDEgMS41Mjk0LTEuNTI5NCAxLjUyOTQgMS41Mjk0IDAgMCAxIDEuNTI5NCAxLjUyOTR6Ii8+CiAgPHBhdGggZD0ibTYuMTM1IDEzLjUzNWMwLTMuMjM5MiAyLjYyNTktNS44NjUgNS44NjUtNS44NjUgMy4yMzkyIDAgNS44NjUgMi42MjU5IDUuODY1IDUuODY1eiIvPgogIDxjaXJjbGUgY3g9IjEyIiBjeT0iMy43Njg1IiByPSIyLjk2ODUiLz4KIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-vega: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbjEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjEyMTIxIj4KICAgIDxwYXRoIGQ9Ik0xMC42IDUuNGwyLjItMy4ySDIuMnY3LjNsNC02LjZ6Ii8+CiAgICA8cGF0aCBkPSJNMTUuOCAyLjJsLTQuNCA2LjZMNyA2LjNsLTQuOCA4djUuNWgxNy42VjIuMmgtNHptLTcgMTUuNEg1LjV2LTQuNGgzLjN2NC40em00LjQgMEg5LjhWOS44aDMuNHY3Ljh6bTQuNCAwaC0zLjRWNi41aDMuNHYxMS4xeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-word: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KIDxnIGNsYXNzPSJqcC1pY29uMiIgZmlsbD0iIzQxNDE0MSI+CiAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiA8L2c+CiA8ZyBjbGFzcz0ianAtaWNvbi1hY2NlbnQyIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSguNDMgLjA0MDEpIiBmaWxsPSIjZmZmIj4KICA8cGF0aCBkPSJtNC4xNCA4Ljc2cTAuMDY4Mi0xLjg5IDIuNDItMS44OSAxLjE2IDAgMS42OCAwLjQyIDAuNTY3IDAuNDEgMC41NjcgMS4xNnYzLjQ3cTAgMC40NjIgMC41MTQgMC40NjIgMC4xMDMgMCAwLjItMC4wMjMxdjAuNzE0cS0wLjM5OSAwLjEwMy0wLjY1MSAwLjEwMy0wLjQ1MiAwLTAuNjkzLTAuMjItMC4yMzEtMC4yLTAuMjg0LTAuNjYyLTAuOTU2IDAuODcyLTIgMC44NzItMC45MDMgMC0xLjQ3LTAuNDcyLTAuNTI1LTAuNDcyLTAuNTI1LTEuMjYgMC0wLjI2MiAwLjA0NTItMC40NzIgMC4wNTY3LTAuMjIgMC4xMTYtMC4zNzggMC4wNjgyLTAuMTY4IDAuMjMxLTAuMzA0IDAuMTU4LTAuMTQ3IDAuMjYyLTAuMjQyIDAuMTE2LTAuMDkxNCAwLjM2OC0wLjE2OCAwLjI2Mi0wLjA5MTQgMC4zOTktMC4xMjYgMC4xMzYtMC4wNDUyIDAuNDcyLTAuMTAzIDAuMzM2LTAuMDU3OCAwLjUwNC0wLjA3OTggMC4xNTgtMC4wMjMxIDAuNTY3LTAuMDc5OCAwLjU1Ni0wLjA2ODIgMC43NzctMC4yMjEgMC4yMi0wLjE1MiAwLjIyLTAuNDQxdi0wLjI1MnEwLTAuNDMtMC4zNTctMC42NjItMC4zMzYtMC4yMzEtMC45NzYtMC4yMzEtMC42NjIgMC0wLjk5OCAwLjI2Mi0wLjMzNiAwLjI1Mi0wLjM5OSAwLjc5OHptMS44OSAzLjY4cTAuNzg4IDAgMS4yNi0wLjQxIDAuNTA0LTAuNDIgMC41MDQtMC45MDN2LTEuMDVxLTAuMjg0IDAuMTM2LTAuODYxIDAuMjMxLTAuNTY3IDAuMDkxNC0wLjk4NyAwLjE1OC0wLjQyIDAuMDY4Mi0wLjc2NiAwLjMyNi0wLjMzNiAwLjI1Mi0wLjMzNiAwLjcwNHQwLjMwNCAwLjcwNCAwLjg2MSAwLjI1MnoiIHN0cm9rZS13aWR0aD0iMS4wNSIvPgogIDxwYXRoIGQ9Im0xMCA0LjU2aDAuOTQ1djMuMTVxMC42NTEtMC45NzYgMS44OS0wLjk3NiAxLjE2IDAgMS44OSAwLjg0IDAuNjgyIDAuODQgMC42ODIgMi4zMSAwIDEuNDctMC43MDQgMi40Mi0wLjcwNCAwLjg4Mi0xLjg5IDAuODgyLTEuMjYgMC0xLjg5LTEuMDJ2MC43NjZoLTAuODV6bTIuNjIgMy4wNHEtMC43NDYgMC0xLjE2IDAuNjQtMC40NTIgMC42My0wLjQ1MiAxLjY4IDAgMS4wNSAwLjQ1MiAxLjY4dDEuMTYgMC42M3EwLjc3NyAwIDEuMjYtMC42MyAwLjQ5NC0wLjY0IDAuNDk0LTEuNjggMC0xLjA1LTAuNDcyLTEuNjgtMC40NjItMC42NC0xLjI2LTAuNjR6IiBzdHJva2Utd2lkdGg9IjEuMDUiLz4KICA8cGF0aCBkPSJtMi43MyAxNS44IDEzLjYgMC4wMDgxYzAuMDA2OSAwIDAtMi42IDAtMi42IDAtMC4wMDc4LTEuMTUgMC0xLjE1IDAtMC4wMDY5IDAtMC4wMDgzIDEuNS0wLjAwODMgMS41LTJlLTMgLTAuMDAxNC0xMS4zLTAuMDAxNC0xMS4zLTAuMDAxNGwtMC4wMDU5Mi0xLjVjMC0wLjAwNzgtMS4xNyAwLjAwMTMtMS4xNyAwLjAwMTN6IiBzdHJva2Utd2lkdGg9Ii45NzUiLz4KIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-yaml: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1jb250cmFzdDIganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjRDgxQjYwIj4KICAgIDxwYXRoIGQ9Ik03LjIgMTguNnYtNS40TDMgNS42aDMuM2wxLjQgMy4xYy4zLjkuNiAxLjYgMSAyLjUuMy0uOC42LTEuNiAxLTIuNWwxLjQtMy4xaDMuNGwtNC40IDcuNnY1LjVsLTIuOS0uMXoiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxNi41IiByPSIyLjEiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxMSIgcj0iMi4xIi8+CiAgPC9nPgo8L3N2Zz4K);
}

/* Icon CSS class declarations */

.jp-AddAboveIcon {
  background-image: var(--jp-icon-add-above);
}

.jp-AddBelowIcon {
  background-image: var(--jp-icon-add-below);
}

.jp-AddIcon {
  background-image: var(--jp-icon-add);
}

.jp-BellIcon {
  background-image: var(--jp-icon-bell);
}

.jp-BugDotIcon {
  background-image: var(--jp-icon-bug-dot);
}

.jp-BugIcon {
  background-image: var(--jp-icon-bug);
}

.jp-BuildIcon {
  background-image: var(--jp-icon-build);
}

.jp-CaretDownEmptyIcon {
  background-image: var(--jp-icon-caret-down-empty);
}

.jp-CaretDownEmptyThinIcon {
  background-image: var(--jp-icon-caret-down-empty-thin);
}

.jp-CaretDownIcon {
  background-image: var(--jp-icon-caret-down);
}

.jp-CaretLeftIcon {
  background-image: var(--jp-icon-caret-left);
}

.jp-CaretRightIcon {
  background-image: var(--jp-icon-caret-right);
}

.jp-CaretUpEmptyThinIcon {
  background-image: var(--jp-icon-caret-up-empty-thin);
}

.jp-CaretUpIcon {
  background-image: var(--jp-icon-caret-up);
}

.jp-CaseSensitiveIcon {
  background-image: var(--jp-icon-case-sensitive);
}

.jp-CheckIcon {
  background-image: var(--jp-icon-check);
}

.jp-CircleEmptyIcon {
  background-image: var(--jp-icon-circle-empty);
}

.jp-CircleIcon {
  background-image: var(--jp-icon-circle);
}

.jp-ClearIcon {
  background-image: var(--jp-icon-clear);
}

.jp-CloseIcon {
  background-image: var(--jp-icon-close);
}

.jp-CodeCheckIcon {
  background-image: var(--jp-icon-code-check);
}

.jp-CodeIcon {
  background-image: var(--jp-icon-code);
}

.jp-CollapseAllIcon {
  background-image: var(--jp-icon-collapse-all);
}

.jp-ConsoleIcon {
  background-image: var(--jp-icon-console);
}

.jp-CopyIcon {
  background-image: var(--jp-icon-copy);
}

.jp-CopyrightIcon {
  background-image: var(--jp-icon-copyright);
}

.jp-CutIcon {
  background-image: var(--jp-icon-cut);
}

.jp-DeleteIcon {
  background-image: var(--jp-icon-delete);
}

.jp-DownloadIcon {
  background-image: var(--jp-icon-download);
}

.jp-DuplicateIcon {
  background-image: var(--jp-icon-duplicate);
}

.jp-EditIcon {
  background-image: var(--jp-icon-edit);
}

.jp-EllipsesIcon {
  background-image: var(--jp-icon-ellipses);
}

.jp-ErrorIcon {
  background-image: var(--jp-icon-error);
}

.jp-ExpandAllIcon {
  background-image: var(--jp-icon-expand-all);
}

.jp-ExtensionIcon {
  background-image: var(--jp-icon-extension);
}

.jp-FastForwardIcon {
  background-image: var(--jp-icon-fast-forward);
}

.jp-FileIcon {
  background-image: var(--jp-icon-file);
}

.jp-FileUploadIcon {
  background-image: var(--jp-icon-file-upload);
}

.jp-FilterDotIcon {
  background-image: var(--jp-icon-filter-dot);
}

.jp-FilterIcon {
  background-image: var(--jp-icon-filter);
}

.jp-FilterListIcon {
  background-image: var(--jp-icon-filter-list);
}

.jp-FolderFavoriteIcon {
  background-image: var(--jp-icon-folder-favorite);
}

.jp-FolderIcon {
  background-image: var(--jp-icon-folder);
}

.jp-HomeIcon {
  background-image: var(--jp-icon-home);
}

.jp-Html5Icon {
  background-image: var(--jp-icon-html5);
}

.jp-ImageIcon {
  background-image: var(--jp-icon-image);
}

.jp-InfoIcon {
  background-image: var(--jp-icon-info);
}

.jp-InspectorIcon {
  background-image: var(--jp-icon-inspector);
}

.jp-JsonIcon {
  background-image: var(--jp-icon-json);
}

.jp-JuliaIcon {
  background-image: var(--jp-icon-julia);
}

.jp-JupyterFaviconIcon {
  background-image: var(--jp-icon-jupyter-favicon);
}

.jp-JupyterIcon {
  background-image: var(--jp-icon-jupyter);
}

.jp-JupyterlabWordmarkIcon {
  background-image: var(--jp-icon-jupyterlab-wordmark);
}

.jp-KernelIcon {
  background-image: var(--jp-icon-kernel);
}

.jp-KeyboardIcon {
  background-image: var(--jp-icon-keyboard);
}

.jp-LaunchIcon {
  background-image: var(--jp-icon-launch);
}

.jp-LauncherIcon {
  background-image: var(--jp-icon-launcher);
}

.jp-LineFormIcon {
  background-image: var(--jp-icon-line-form);
}

.jp-LinkIcon {
  background-image: var(--jp-icon-link);
}

.jp-ListIcon {
  background-image: var(--jp-icon-list);
}

.jp-MarkdownIcon {
  background-image: var(--jp-icon-markdown);
}

.jp-MoveDownIcon {
  background-image: var(--jp-icon-move-down);
}

.jp-MoveUpIcon {
  background-image: var(--jp-icon-move-up);
}

.jp-NewFolderIcon {
  background-image: var(--jp-icon-new-folder);
}

.jp-NotTrustedIcon {
  background-image: var(--jp-icon-not-trusted);
}

.jp-NotebookIcon {
  background-image: var(--jp-icon-notebook);
}

.jp-NumberingIcon {
  background-image: var(--jp-icon-numbering);
}

.jp-OfflineBoltIcon {
  background-image: var(--jp-icon-offline-bolt);
}

.jp-PaletteIcon {
  background-image: var(--jp-icon-palette);
}

.jp-PasteIcon {
  background-image: var(--jp-icon-paste);
}

.jp-PdfIcon {
  background-image: var(--jp-icon-pdf);
}

.jp-PythonIcon {
  background-image: var(--jp-icon-python);
}

.jp-RKernelIcon {
  background-image: var(--jp-icon-r-kernel);
}

.jp-ReactIcon {
  background-image: var(--jp-icon-react);
}

.jp-RedoIcon {
  background-image: var(--jp-icon-redo);
}

.jp-RefreshIcon {
  background-image: var(--jp-icon-refresh);
}

.jp-RegexIcon {
  background-image: var(--jp-icon-regex);
}

.jp-RunIcon {
  background-image: var(--jp-icon-run);
}

.jp-RunningIcon {
  background-image: var(--jp-icon-running);
}

.jp-SaveIcon {
  background-image: var(--jp-icon-save);
}

.jp-SearchIcon {
  background-image: var(--jp-icon-search);
}

.jp-SettingsIcon {
  background-image: var(--jp-icon-settings);
}

.jp-ShareIcon {
  background-image: var(--jp-icon-share);
}

.jp-SpreadsheetIcon {
  background-image: var(--jp-icon-spreadsheet);
}

.jp-StopIcon {
  background-image: var(--jp-icon-stop);
}

.jp-TabIcon {
  background-image: var(--jp-icon-tab);
}

.jp-TableRowsIcon {
  background-image: var(--jp-icon-table-rows);
}

.jp-TagIcon {
  background-image: var(--jp-icon-tag);
}

.jp-TerminalIcon {
  background-image: var(--jp-icon-terminal);
}

.jp-TextEditorIcon {
  background-image: var(--jp-icon-text-editor);
}

.jp-TocIcon {
  background-image: var(--jp-icon-toc);
}

.jp-TreeViewIcon {
  background-image: var(--jp-icon-tree-view);
}

.jp-TrustedIcon {
  background-image: var(--jp-icon-trusted);
}

.jp-UndoIcon {
  background-image: var(--jp-icon-undo);
}

.jp-UserIcon {
  background-image: var(--jp-icon-user);
}

.jp-UsersIcon {
  background-image: var(--jp-icon-users);
}

.jp-VegaIcon {
  background-image: var(--jp-icon-vega);
}

.jp-WordIcon {
  background-image: var(--jp-icon-word);
}

.jp-YamlIcon {
  background-image: var(--jp-icon-yaml);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

.jp-Icon,
.jp-MaterialIcon {
  background-position: center;
  background-repeat: no-repeat;
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-cover {
  background-position: center;
  background-repeat: no-repeat;
  background-size: cover;
}

/**
 * (DEPRECATED) Support for specific CSS icon sizes
 */

.jp-Icon-16 {
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-18 {
  background-size: 18px;
  min-width: 18px;
  min-height: 18px;
}

.jp-Icon-20 {
  background-size: 20px;
  min-width: 20px;
  min-height: 20px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.lm-TabBar .lm-TabBar-addButton {
  align-items: center;
  display: flex;
  padding: 4px;
  padding-bottom: 5px;
  margin-right: 1px;
  background-color: var(--jp-layout-color2);
}

.lm-TabBar .lm-TabBar-addButton:hover {
  background-color: var(--jp-layout-color1);
}

.lm-DockPanel-tabBar .lm-TabBar-tab {
  width: var(--jp-private-horizontal-tab-width);
}

.lm-DockPanel-tabBar .lm-TabBar-content {
  flex: unset;
}

.lm-DockPanel-tabBar[data-orientation='horizontal'] {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for icons as inline SVG HTMLElements
 */

/* recolor the primary elements of an icon */
.jp-icon0[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon1[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon2[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon3[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/* recolor the accent elements of an icon */
.jp-icon-accent0[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-accent1[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-accent2[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-accent3[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-accent4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-accent0[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-accent1[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-accent2[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-accent3[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-accent4[stroke] {
  stroke: var(--jp-layout-color4);
}

/* set the color of an icon to transparent */
.jp-icon-none[fill] {
  fill: none;
}

.jp-icon-none[stroke] {
  stroke: none;
}

/* brand icon colors. Same for light and dark */
.jp-icon-brand0[fill] {
  fill: var(--jp-brand-color0);
}

.jp-icon-brand1[fill] {
  fill: var(--jp-brand-color1);
}

.jp-icon-brand2[fill] {
  fill: var(--jp-brand-color2);
}

.jp-icon-brand3[fill] {
  fill: var(--jp-brand-color3);
}

.jp-icon-brand4[fill] {
  fill: var(--jp-brand-color4);
}

.jp-icon-brand0[stroke] {
  stroke: var(--jp-brand-color0);
}

.jp-icon-brand1[stroke] {
  stroke: var(--jp-brand-color1);
}

.jp-icon-brand2[stroke] {
  stroke: var(--jp-brand-color2);
}

.jp-icon-brand3[stroke] {
  stroke: var(--jp-brand-color3);
}

.jp-icon-brand4[stroke] {
  stroke: var(--jp-brand-color4);
}

/* warn icon colors. Same for light and dark */
.jp-icon-warn0[fill] {
  fill: var(--jp-warn-color0);
}

.jp-icon-warn1[fill] {
  fill: var(--jp-warn-color1);
}

.jp-icon-warn2[fill] {
  fill: var(--jp-warn-color2);
}

.jp-icon-warn3[fill] {
  fill: var(--jp-warn-color3);
}

.jp-icon-warn0[stroke] {
  stroke: var(--jp-warn-color0);
}

.jp-icon-warn1[stroke] {
  stroke: var(--jp-warn-color1);
}

.jp-icon-warn2[stroke] {
  stroke: var(--jp-warn-color2);
}

.jp-icon-warn3[stroke] {
  stroke: var(--jp-warn-color3);
}

/* icon colors that contrast well with each other and most backgrounds */
.jp-icon-contrast0[fill] {
  fill: var(--jp-icon-contrast-color0);
}

.jp-icon-contrast1[fill] {
  fill: var(--jp-icon-contrast-color1);
}

.jp-icon-contrast2[fill] {
  fill: var(--jp-icon-contrast-color2);
}

.jp-icon-contrast3[fill] {
  fill: var(--jp-icon-contrast-color3);
}

.jp-icon-contrast0[stroke] {
  stroke: var(--jp-icon-contrast-color0);
}

.jp-icon-contrast1[stroke] {
  stroke: var(--jp-icon-contrast-color1);
}

.jp-icon-contrast2[stroke] {
  stroke: var(--jp-icon-contrast-color2);
}

.jp-icon-contrast3[stroke] {
  stroke: var(--jp-icon-contrast-color3);
}

.jp-icon-dot[fill] {
  fill: var(--jp-warn-color0);
}

.jp-jupyter-icon-color[fill] {
  fill: var(--jp-jupyter-icon-color, var(--jp-warn-color0));
}

.jp-notebook-icon-color[fill] {
  fill: var(--jp-notebook-icon-color, var(--jp-warn-color0));
}

.jp-json-icon-color[fill] {
  fill: var(--jp-json-icon-color, var(--jp-warn-color1));
}

.jp-console-icon-color[fill] {
  fill: var(--jp-console-icon-color, white);
}

.jp-console-icon-background-color[fill] {
  fill: var(--jp-console-icon-background-color, var(--jp-brand-color1));
}

.jp-terminal-icon-color[fill] {
  fill: var(--jp-terminal-icon-color, var(--jp-layout-color2));
}

.jp-terminal-icon-background-color[fill] {
  fill: var(
    --jp-terminal-icon-background-color,
    var(--jp-inverse-layout-color2)
  );
}

.jp-text-editor-icon-color[fill] {
  fill: var(--jp-text-editor-icon-color, var(--jp-inverse-layout-color3));
}

.jp-inspector-icon-color[fill] {
  fill: var(--jp-inspector-icon-color, var(--jp-inverse-layout-color3));
}

/* CSS for icons in selected filebrowser listing items */
.jp-DirListing-item.jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}

.jp-DirListing-item.jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* stylelint-disable selector-max-class, selector-max-compound-selectors */

/**
* TODO: come up with non css-hack solution for showing the busy icon on top
*  of the close icon
* CSS for complex behavior of close icon of tabs in the main area tabbar
*/
.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon3[fill] {
  fill: none;
}

.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon-busy[fill] {
  fill: var(--jp-inverse-layout-color3);
}

/* stylelint-enable selector-max-class, selector-max-compound-selectors */

/* CSS for icons in status bar */
#jp-main-statusbar .jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}

#jp-main-statusbar .jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* special handling for splash icon CSS. While the theme CSS reloads during
   splash, the splash icon can loose theming. To prevent that, we set a
   default for its color variable */
:root {
  --jp-warn-color0: var(--md-orange-700);
}

/* not sure what to do with this one, used in filebrowser listing */
.jp-DragIcon {
  margin-right: 4px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for alt colors for icons as inline SVG HTMLElements
 */

/* alt recolor the primary elements of an icon */
.jp-icon-alt .jp-icon0[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-alt .jp-icon1[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-alt .jp-icon2[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-alt .jp-icon3[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-alt .jp-icon4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-alt .jp-icon0[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-alt .jp-icon1[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-alt .jp-icon2[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-alt .jp-icon3[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-alt .jp-icon4[stroke] {
  stroke: var(--jp-layout-color4);
}

/* alt recolor the accent elements of an icon */
.jp-icon-alt .jp-icon-accent0[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon-alt .jp-icon-accent1[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon-alt .jp-icon-accent2[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon-alt .jp-icon-accent3[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon-alt .jp-icon-accent4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-alt .jp-icon-accent0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon-alt .jp-icon-accent1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon-alt .jp-icon-accent2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon-alt .jp-icon-accent3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon-alt .jp-icon-accent4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-icon-hoverShow:not(:hover) .jp-icon-hoverShow-content {
  display: none !important;
}

/**
 * Support for hover colors for icons as inline SVG HTMLElements
 */

/**
 * regular colors
 */

/* recolor the primary elements of an icon */
.jp-icon-hover :hover .jp-icon0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon-hover :hover .jp-icon1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon-hover :hover .jp-icon2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon-hover :hover .jp-icon3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon-hover :hover .jp-icon4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon-hover :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon-hover :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon-hover :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon-hover :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/* recolor the accent elements of an icon */
.jp-icon-hover :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-hover :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-hover :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-hover :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-hover :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-hover :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-hover :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-hover :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-hover :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* set the color of an icon to transparent */
.jp-icon-hover :hover .jp-icon-none-hover[fill] {
  fill: none;
}

.jp-icon-hover :hover .jp-icon-none-hover[stroke] {
  stroke: none;
}

/**
 * inverse colors
 */

/* inverse recolor the primary elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* inverse recolor the accent elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-IFrame {
  width: 100%;
  height: 100%;
}

.jp-IFrame > iframe {
  border: none;
}

/*
When drag events occur, `lm-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-IFrame {
  position: relative;
}

body.lm-mod-override-cursor .jp-IFrame::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-HoverBox {
  position: fixed;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-FormGroup-content fieldset {
  border: none;
  padding: 0;
  min-width: 0;
  width: 100%;
}

/* stylelint-disable selector-max-type */

.jp-FormGroup-content fieldset .jp-inputFieldWrapper input,
.jp-FormGroup-content fieldset .jp-inputFieldWrapper select,
.jp-FormGroup-content fieldset .jp-inputFieldWrapper textarea {
  font-size: var(--jp-content-font-size2);
  border-color: var(--jp-input-border-color);
  border-style: solid;
  border-radius: var(--jp-border-radius);
  border-width: 1px;
  padding: 6px 8px;
  background: none;
  color: var(--jp-ui-font-color0);
  height: inherit;
}

.jp-FormGroup-content fieldset input[type='checkbox'] {
  position: relative;
  top: 2px;
  margin-left: 0;
}

.jp-FormGroup-content button.jp-mod-styled {
  cursor: pointer;
}

.jp-FormGroup-content .checkbox label {
  cursor: pointer;
  font-size: var(--jp-content-font-size1);
}

.jp-FormGroup-content .jp-root > fieldset > legend {
  display: none;
}

.jp-FormGroup-content .jp-root > fieldset > p {
  display: none;
}

/** copy of `input.jp-mod-styled:focus` style */
.jp-FormGroup-content fieldset input:focus,
.jp-FormGroup-content fieldset select:focus {
  -moz-outline-radius: unset;
  outline: var(--jp-border-width) solid var(--md-blue-500);
  outline-offset: -1px;
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-FormGroup-content fieldset input:hover:not(:focus),
.jp-FormGroup-content fieldset select:hover:not(:focus) {
  background-color: var(--jp-border-color2);
}

/* stylelint-enable selector-max-type */

.jp-FormGroup-content .checkbox .field-description {
  /* Disable default description field for checkbox:
   because other widgets do not have description fields,
   we add descriptions to each widget on the field level.
  */
  display: none;
}

.jp-FormGroup-content #root__description {
  display: none;
}

.jp-FormGroup-content .jp-modifiedIndicator {
  width: 5px;
  background-color: var(--jp-brand-color2);
  margin-top: 0;
  margin-left: calc(var(--jp-private-settingeditor-modifier-indent) * -1);
  flex-shrink: 0;
}

.jp-FormGroup-content .jp-modifiedIndicator.jp-errorIndicator {
  background-color: var(--jp-error-color0);
  margin-right: 0.5em;
}

/* RJSF ARRAY style */

.jp-arrayFieldWrapper legend {
  font-size: var(--jp-content-font-size2);
  color: var(--jp-ui-font-color0);
  flex-basis: 100%;
  padding: 4px 0;
  font-weight: var(--jp-content-heading-font-weight);
  border-bottom: 1px solid var(--jp-border-color2);
}

.jp-arrayFieldWrapper .field-description {
  padding: 4px 0;
  white-space: pre-wrap;
}

.jp-arrayFieldWrapper .array-item {
  width: 100%;
  border: 1px solid var(--jp-border-color2);
  border-radius: 4px;
  margin: 4px;
}

.jp-ArrayOperations {
  display: flex;
  margin-left: 8px;
}

.jp-ArrayOperationsButton {
  margin: 2px;
}

.jp-ArrayOperationsButton .jp-icon3[fill] {
  fill: var(--jp-ui-font-color0);
}

button.jp-ArrayOperationsButton.jp-mod-styled:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

/* RJSF form validation error */

.jp-FormGroup-content .validationErrors {
  color: var(--jp-error-color0);
}

/* Hide panel level error as duplicated the field level error */
.jp-FormGroup-content .panel.errors {
  display: none;
}

/* RJSF normal content (settings-editor) */

.jp-FormGroup-contentNormal {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
}

.jp-FormGroup-contentNormal .jp-FormGroup-contentItem {
  margin-left: 7px;
  color: var(--jp-ui-font-color0);
}

.jp-FormGroup-contentNormal .jp-FormGroup-description {
  flex-basis: 100%;
  padding: 4px 7px;
}

.jp-FormGroup-contentNormal .jp-FormGroup-default {
  flex-basis: 100%;
  padding: 4px 7px;
}

.jp-FormGroup-contentNormal .jp-FormGroup-fieldLabel {
  font-size: var(--jp-content-font-size1);
  font-weight: normal;
  min-width: 120px;
}

.jp-FormGroup-contentNormal fieldset:not(:first-child) {
  margin-left: 7px;
}

.jp-FormGroup-contentNormal .field-array-of-string .array-item {
  /* Display `jp-ArrayOperations` buttons side-by-side with content except
    for small screens where flex-wrap will place them one below the other.
  */
  display: flex;
  align-items: center;
  flex-wrap: wrap;
}

.jp-FormGroup-contentNormal .jp-objectFieldWrapper .form-group {
  padding: 2px 8px 2px var(--jp-private-settingeditor-modifier-indent);
  margin-top: 2px;
}

/* RJSF compact content (metadata-form) */

.jp-FormGroup-content.jp-FormGroup-contentCompact {
  width: 100%;
}

.jp-FormGroup-contentCompact .form-group {
  display: flex;
  padding: 0.5em 0.2em 0.5em 0;
}

.jp-FormGroup-contentCompact
  .jp-FormGroup-compactTitle
  .jp-FormGroup-description {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color2);
}

.jp-FormGroup-contentCompact .jp-FormGroup-fieldLabel {
  padding-bottom: 0.3em;
}

.jp-FormGroup-contentCompact .jp-inputFieldWrapper .form-control {
  width: 100%;
  box-sizing: border-box;
}

.jp-FormGroup-contentCompact .jp-arrayFieldWrapper .jp-FormGroup-compactTitle {
  padding-bottom: 7px;
}

.jp-FormGroup-contentCompact
  .jp-objectFieldWrapper
  .jp-objectFieldWrapper
  .form-group {
  padding: 2px 8px 2px var(--jp-private-settingeditor-modifier-indent);
  margin-top: 2px;
}

.jp-FormGroup-contentCompact ul.error-detail {
  margin-block-start: 0.5em;
  margin-block-end: 0.5em;
  padding-inline-start: 1em;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-SidePanel {
  display: flex;
  flex-direction: column;
  min-width: var(--jp-sidebar-min-width);
  overflow-y: auto;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  font-size: var(--jp-ui-font-size1);
}

.jp-SidePanel-header {
  flex: 0 0 auto;
  display: flex;
  border-bottom: var(--jp-border-width) solid var(--jp-border-color2);
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  letter-spacing: 1px;
  margin: 0;
  padding: 2px;
  text-transform: uppercase;
}

.jp-SidePanel-toolbar {
  flex: 0 0 auto;
}

.jp-SidePanel-content {
  flex: 1 1 auto;
}

.jp-SidePanel-toolbar,
.jp-AccordionPanel-toolbar {
  height: var(--jp-private-toolbar-height);
}

.jp-SidePanel-toolbar.jp-Toolbar-micro {
  display: none;
}

.lm-AccordionPanel .jp-AccordionPanel-title {
  box-sizing: border-box;
  line-height: 25px;
  margin: 0;
  display: flex;
  align-items: center;
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  font-size: var(--jp-ui-font-size0);
}

.jp-AccordionPanel-title {
  cursor: pointer;
  user-select: none;
  -moz-user-select: none;
  -webkit-user-select: none;
  text-transform: uppercase;
}

.lm-AccordionPanel[data-orientation='horizontal'] > .jp-AccordionPanel-title {
  /* Title is rotated for horizontal accordion panel using CSS */
  display: block;
  transform-origin: top left;
  transform: rotate(-90deg) translate(-100%);
}

.jp-AccordionPanel-title .lm-AccordionPanel-titleLabel {
  user-select: none;
  text-overflow: ellipsis;
  white-space: nowrap;
  overflow: hidden;
}

.jp-AccordionPanel-title .lm-AccordionPanel-titleCollapser {
  transform: rotate(-90deg);
  margin: auto 0;
  height: 16px;
}

.jp-AccordionPanel-title.lm-mod-expanded .lm-AccordionPanel-titleCollapser {
  transform: rotate(0deg);
}

.lm-AccordionPanel .jp-AccordionPanel-toolbar {
  background: none;
  box-shadow: none;
  border: none;
  margin-left: auto;
}

.lm-AccordionPanel .lm-SplitPanel-handle:hover {
  background: var(--jp-layout-color3);
}

.jp-text-truncated {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Spinner {
  position: absolute;
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 10;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-layout-color0);
  outline: none;
}

.jp-SpinnerContent {
  font-size: 10px;
  margin: 50px auto;
  text-indent: -9999em;
  width: 3em;
  height: 3em;
  border-radius: 50%;
  background: var(--jp-brand-color3);
  background: linear-gradient(
    to right,
    #f37626 10%,
    rgba(255, 255, 255, 0) 42%
  );
  position: relative;
  animation: load3 1s infinite linear, fadeIn 1s;
}

.jp-SpinnerContent::before {
  width: 50%;
  height: 50%;
  background: #f37626;
  border-radius: 100% 0 0;
  position: absolute;
  top: 0;
  left: 0;
  content: '';
}

.jp-SpinnerContent::after {
  background: var(--jp-layout-color0);
  width: 75%;
  height: 75%;
  border-radius: 50%;
  content: '';
  margin: auto;
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
}

@keyframes fadeIn {
  0% {
    opacity: 0;
  }

  100% {
    opacity: 1;
  }
}

@keyframes load3 {
  0% {
    transform: rotate(0deg);
  }

  100% {
    transform: rotate(360deg);
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

button.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: none;
  box-sizing: border-box;
  text-align: center;
  line-height: 32px;
  height: 32px;
  padding: 0 12px;
  letter-spacing: 0.8px;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input.jp-mod-styled {
  background: var(--jp-input-background);
  height: 28px;
  box-sizing: border-box;
  border: var(--jp-border-width) solid var(--jp-border-color1);
  padding-left: 7px;
  padding-right: 7px;
  font-size: var(--jp-ui-font-size2);
  color: var(--jp-ui-font-color0);
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input[type='checkbox'].jp-mod-styled {
  appearance: checkbox;
  -webkit-appearance: checkbox;
  -moz-appearance: checkbox;
  height: auto;
}

input.jp-mod-styled:focus {
  border: var(--jp-border-width) solid var(--md-blue-500);
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-select-wrapper {
  display: flex;
  position: relative;
  flex-direction: column;
  padding: 1px;
  background-color: var(--jp-layout-color1);
  box-sizing: border-box;
  margin-bottom: 12px;
}

.jp-select-wrapper:not(.multiple) {
  height: 28px;
}

.jp-select-wrapper.jp-mod-focused select.jp-mod-styled {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-input-active-background);
}

select.jp-mod-styled:hover {
  cursor: pointer;
  color: var(--jp-ui-font-color0);
  background-color: var(--jp-input-hover-background);
  box-shadow: inset 0 0 1px rgba(0, 0, 0, 0.5);
}

select.jp-mod-styled {
  flex: 1 1 auto;
  width: 100%;
  font-size: var(--jp-ui-font-size2);
  background: var(--jp-input-background);
  color: var(--jp-ui-font-color0);
  padding: 0 25px 0 8px;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

select.jp-mod-styled:not([multiple]) {
  height: 32px;
}

select.jp-mod-styled[multiple] {
  max-height: 200px;
  overflow-y: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-switch {
  display: flex;
  align-items: center;
  padding-left: 4px;
  padding-right: 4px;
  font-size: var(--jp-ui-font-size1);
  background-color: transparent;
  color: var(--jp-ui-font-color1);
  border: none;
  height: 20px;
}

.jp-switch:hover {
  background-color: var(--jp-layout-color2);
}

.jp-switch-label {
  margin-right: 5px;
  font-family: var(--jp-ui-font-family);
}

.jp-switch-track {
  cursor: pointer;
  background-color: var(--jp-switch-color, var(--jp-border-color1));
  -webkit-transition: 0.4s;
  transition: 0.4s;
  border-radius: 34px;
  height: 16px;
  width: 35px;
  position: relative;
}

.jp-switch-track::before {
  content: '';
  position: absolute;
  height: 10px;
  width: 10px;
  margin: 3px;
  left: 0;
  background-color: var(--jp-ui-inverse-font-color1);
  -webkit-transition: 0.4s;
  transition: 0.4s;
  border-radius: 50%;
}

.jp-switch[aria-checked='true'] .jp-switch-track {
  background-color: var(--jp-switch-true-position-color, var(--jp-warn-color0));
}

.jp-switch[aria-checked='true'] .jp-switch-track::before {
  /* track width (35) - margins (3 + 3) - thumb width (10) */
  left: 19px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

:root {
  --jp-private-toolbar-height: calc(
    28px + var(--jp-border-width)
  ); /* leave 28px for content */
}

.jp-Toolbar {
  color: var(--jp-ui-font-color1);
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  background: var(--jp-toolbar-background);
  min-height: var(--jp-toolbar-micro-height);
  padding: 2px;
  z-index: 8;
  overflow-x: hidden;
}

/* Toolbar items */

.jp-Toolbar > .jp-Toolbar-item.jp-Toolbar-spacer {
  flex-grow: 1;
  flex-shrink: 1;
}

.jp-Toolbar-item.jp-Toolbar-kernelStatus {
  display: inline-block;
  width: 32px;
  background-repeat: no-repeat;
  background-position: center;
  background-size: 16px;
}

.jp-Toolbar > .jp-Toolbar-item {
  flex: 0 0 auto;
  display: flex;
  padding-left: 1px;
  padding-right: 1px;
  font-size: var(--jp-ui-font-size1);
  line-height: var(--jp-private-toolbar-height);
  height: 100%;
}

/* Toolbar buttons */

/* This is the div we use to wrap the react component into a Widget */
div.jp-ToolbarButton {
  color: transparent;
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0;
  margin: 0;
}

button.jp-ToolbarButtonComponent {
  background: var(--jp-layout-color1);
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0 6px;
  margin: 0;
  height: 24px;
  border-radius: var(--jp-border-radius);
  display: flex;
  align-items: center;
  text-align: center;
  font-size: 14px;
  min-width: unset;
  min-height: unset;
}

button.jp-ToolbarButtonComponent:disabled {
  opacity: 0.4;
}

button.jp-ToolbarButtonComponent > span {
  padding: 0;
  flex: 0 0 auto;
}

button.jp-ToolbarButtonComponent .jp-ToolbarButtonComponent-label {
  font-size: var(--jp-ui-font-size1);
  line-height: 100%;
  padding-left: 2px;
  color: var(--jp-ui-font-color1);
  font-family: var(--jp-ui-font-family);
}

#jp-main-dock-panel[data-mode='single-document']
  .jp-MainAreaWidget
  > .jp-Toolbar.jp-Toolbar-micro {
  padding: 0;
  min-height: 0;
}

#jp-main-dock-panel[data-mode='single-document']
  .jp-MainAreaWidget
  > .jp-Toolbar {
  border: none;
  box-shadow: none;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-WindowedPanel-outer {
  position: relative;
  overflow-y: auto;
}

.jp-WindowedPanel-inner {
  position: relative;
}

.jp-WindowedPanel-window {
  position: absolute;
  left: 0;
  right: 0;
  overflow: visible;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* Sibling imports */

body {
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
}

/* Disable native link decoration styles everywhere outside of dialog boxes */
a {
  text-decoration: unset;
  color: unset;
}

a:hover {
  text-decoration: unset;
  color: unset;
}

/* Accessibility for links inside dialog box text */
.jp-Dialog-content a {
  text-decoration: revert;
  color: var(--jp-content-link-color);
}

.jp-Dialog-content a:hover {
  text-decoration: revert;
}

/* Styles for ui-components */
.jp-Button {
  color: var(--jp-ui-font-color2);
  border-radius: var(--jp-border-radius);
  padding: 0 12px;
  font-size: var(--jp-ui-font-size1);

  /* Copy from blueprint 3 */
  display: inline-flex;
  flex-direction: row;
  border: none;
  cursor: pointer;
  align-items: center;
  justify-content: center;
  text-align: left;
  vertical-align: middle;
  min-height: 30px;
  min-width: 30px;
}

.jp-Button:disabled {
  cursor: not-allowed;
}

.jp-Button:empty {
  padding: 0 !important;
}

.jp-Button.jp-mod-small {
  min-height: 24px;
  min-width: 24px;
  font-size: 12px;
  padding: 0 7px;
}

/* Use our own theme for hover styles */
.jp-Button.jp-mod-minimal:hover {
  background-color: var(--jp-layout-color2);
}

.jp-Button.jp-mod-minimal {
  background: none;
}

.jp-InputGroup {
  display: block;
  position: relative;
}

.jp-InputGroup input {
  box-sizing: border-box;
  border: none;
  border-radius: 0;
  background-color: transparent;
  color: var(--jp-ui-font-color0);
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
  padding-bottom: 0;
  padding-top: 0;
  padding-left: 10px;
  padding-right: 28px;
  position: relative;
  width: 100%;
  -webkit-appearance: none;
  -moz-appearance: none;
  appearance: none;
  font-size: 14px;
  font-weight: 400;
  height: 30px;
  line-height: 30px;
  outline: none;
  vertical-align: middle;
}

.jp-InputGroup input:focus {
  box-shadow: inset 0 0 0 var(--jp-border-width)
      var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.jp-InputGroup input:disabled {
  cursor: not-allowed;
  resize: block;
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color2);
}

.jp-InputGroup input:disabled ~ span {
  cursor: not-allowed;
  color: var(--jp-ui-font-color2);
}

.jp-InputGroup input::placeholder,
input::placeholder {
  color: var(--jp-ui-font-color2);
}

.jp-InputGroupAction {
  position: absolute;
  bottom: 1px;
  right: 0;
  padding: 6px;
}

.jp-HTMLSelect.jp-DefaultStyle select {
  background-color: initial;
  border: none;
  border-radius: 0;
  box-shadow: none;
  color: var(--jp-ui-font-color0);
  display: block;
  font-size: var(--jp-ui-font-size1);
  font-family: var(--jp-ui-font-family);
  height: 24px;
  line-height: 14px;
  padding: 0 25px 0 10px;
  text-align: left;
  -moz-appearance: none;
  -webkit-appearance: none;
}

.jp-HTMLSelect.jp-DefaultStyle select:disabled {
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color2);
  cursor: not-allowed;
  resize: block;
}

.jp-HTMLSelect.jp-DefaultStyle select:disabled ~ span {
  cursor: not-allowed;
}

/* Use our own theme for hover and option styles */
/* stylelint-disable-next-line selector-max-type */
.jp-HTMLSelect.jp-DefaultStyle select:hover,
.jp-HTMLSelect.jp-DefaultStyle select > option {
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color0);
}

select {
  box-sizing: border-box;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-StatusBar-Widget {
  display: flex;
  align-items: center;
  background: var(--jp-layout-color2);
  min-height: var(--jp-statusbar-height);
  justify-content: space-between;
  padding: 0 10px;
}

.jp-StatusBar-Left {
  display: flex;
  align-items: center;
  flex-direction: row;
}

.jp-StatusBar-Middle {
  display: flex;
  align-items: center;
}

.jp-StatusBar-Right {
  display: flex;
  align-items: center;
  flex-direction: row-reverse;
}

.jp-StatusBar-Item {
  max-height: var(--jp-statusbar-height);
  margin: 0 2px;
  height: var(--jp-statusbar-height);
  white-space: nowrap;
  text-overflow: ellipsis;
  color: var(--jp-ui-font-color1);
  padding: 0 6px;
}

.jp-mod-highlighted:hover {
  background-color: var(--jp-layout-color3);
}

.jp-mod-clicked {
  background-color: var(--jp-brand-color1);
}

.jp-mod-clicked:hover {
  background-color: var(--jp-brand-color0);
}

.jp-mod-clicked .jp-StatusBar-TextItem {
  color: var(--jp-ui-inverse-font-color1);
}

.jp-StatusBar-HoverItem {
  box-shadow: '0px 4px 4px rgba(0, 0, 0, 0.25)';
}

.jp-StatusBar-TextItem {
  font-size: var(--jp-ui-font-size1);
  font-family: var(--jp-ui-font-family);
  line-height: 24px;
  color: var(--jp-ui-font-color1);
}

.jp-StatusBar-GroupItem {
  display: flex;
  align-items: center;
  flex-direction: row;
}

.jp-Statusbar-ProgressCircle svg {
  display: block;
  margin: 0 auto;
  width: 16px;
  height: 24px;
  align-self: normal;
}

.jp-Statusbar-ProgressCircle path {
  fill: var(--jp-inverse-layout-color3);
}

.jp-Statusbar-ProgressBar-progress-bar {
  height: 10px;
  width: 100px;
  border: solid 0.25px var(--jp-brand-color2);
  border-radius: 3px;
  overflow: hidden;
  align-self: center;
}

.jp-Statusbar-ProgressBar-progress-bar > div {
  background-color: var(--jp-brand-color2);
  background-image: linear-gradient(
    -45deg,
    rgba(255, 255, 255, 0.2) 25%,
    transparent 25%,
    transparent 50%,
    rgba(255, 255, 255, 0.2) 50%,
    rgba(255, 255, 255, 0.2) 75%,
    transparent 75%,
    transparent
  );
  background-size: 40px 40px;
  float: left;
  width: 0%;
  height: 100%;
  font-size: 12px;
  line-height: 14px;
  color: #fff;
  text-align: center;
  animation: jp-Statusbar-ExecutionTime-progress-bar 2s linear infinite;
}

.jp-Statusbar-ProgressBar-progress-bar p {
  color: var(--jp-ui-font-color1);
  font-family: var(--jp-ui-font-family);
  font-size: var(--jp-ui-font-size1);
  line-height: 10px;
  width: 100px;
}

@keyframes jp-Statusbar-ExecutionTime-progress-bar {
  0% {
    background-position: 0 0;
  }

  100% {
    background-position: 40px 40px;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-commandpalette-search-height: 28px;
}

/*-----------------------------------------------------------------------------
| Overall styles
|----------------------------------------------------------------------------*/

.lm-CommandPalette {
  padding-bottom: 0;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);

  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Modal variant
|----------------------------------------------------------------------------*/

.jp-ModalCommandPalette {
  position: absolute;
  z-index: 10000;
  top: 38px;
  left: 30%;
  margin: 0;
  padding: 4px;
  width: 40%;
  box-shadow: var(--jp-elevation-z4);
  border-radius: 4px;
  background: var(--jp-layout-color0);
}

.jp-ModalCommandPalette .lm-CommandPalette {
  max-height: 40vh;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-close-icon::after {
  display: none;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-CommandPalette-header {
  display: none;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-CommandPalette-item {
  margin-left: 4px;
  margin-right: 4px;
}

.jp-ModalCommandPalette
  .lm-CommandPalette
  .lm-CommandPalette-item.lm-mod-disabled {
  display: none;
}

/*-----------------------------------------------------------------------------
| Search
|----------------------------------------------------------------------------*/

.lm-CommandPalette-search {
  padding: 4px;
  background-color: var(--jp-layout-color1);
  z-index: 2;
}

.lm-CommandPalette-wrapper {
  overflow: overlay;
  padding: 0 9px;
  background-color: var(--jp-input-active-background);
  height: 30px;
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
}

.lm-CommandPalette.lm-mod-focused .lm-CommandPalette-wrapper {
  box-shadow: inset 0 0 0 1px var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.jp-SearchIconGroup {
  color: white;
  background-color: var(--jp-brand-color1);
  position: absolute;
  top: 4px;
  right: 4px;
  padding: 5px 5px 1px;
}

.jp-SearchIconGroup svg {
  height: 20px;
  width: 20px;
}

.jp-SearchIconGroup .jp-icon3[fill] {
  fill: var(--jp-layout-color0);
}

.lm-CommandPalette-input {
  background: transparent;
  width: calc(100% - 18px);
  float: left;
  border: none;
  outline: none;
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  line-height: var(--jp-private-commandpalette-search-height);
}

.lm-CommandPalette-input::-webkit-input-placeholder,
.lm-CommandPalette-input::-moz-placeholder,
.lm-CommandPalette-input:-ms-input-placeholder {
  color: var(--jp-ui-font-color2);
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Results
|----------------------------------------------------------------------------*/

.lm-CommandPalette-header:first-child {
  margin-top: 0;
}

.lm-CommandPalette-header {
  border-bottom: solid var(--jp-border-width) var(--jp-border-color2);
  color: var(--jp-ui-font-color1);
  cursor: pointer;
  display: flex;
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  letter-spacing: 1px;
  margin-top: 8px;
  padding: 8px 0 8px 12px;
  text-transform: uppercase;
}

.lm-CommandPalette-header.lm-mod-active {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-header > mark {
  background-color: transparent;
  font-weight: bold;
  color: var(--jp-ui-font-color1);
}

.lm-CommandPalette-item {
  padding: 4px 12px 4px 4px;
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  font-weight: 400;
  display: flex;
}

.lm-CommandPalette-item.lm-mod-disabled {
  color: var(--jp-ui-font-color2);
}

.lm-CommandPalette-item.lm-mod-active {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.lm-CommandPalette-item.lm-mod-active .lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-inverse-font-color0);
}

.lm-CommandPalette-item.lm-mod-active .jp-icon-selectable[fill] {
  fill: var(--jp-layout-color0);
}

.lm-CommandPalette-item.lm-mod-active:hover:not(.lm-mod-disabled) {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.lm-CommandPalette-item:hover:not(.lm-mod-active):not(.lm-mod-disabled) {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-itemContent {
  overflow: hidden;
}

.lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-font-color0);
  background-color: transparent;
  font-weight: bold;
}

.lm-CommandPalette-item.lm-mod-disabled mark {
  color: var(--jp-ui-font-color2);
}

.lm-CommandPalette-item .lm-CommandPalette-itemIcon {
  margin: 0 4px 0 0;
  position: relative;
  width: 16px;
  top: 2px;
  flex: 0 0 auto;
}

.lm-CommandPalette-item.lm-mod-disabled .lm-CommandPalette-itemIcon {
  opacity: 0.6;
}

.lm-CommandPalette-item .lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemCaption {
  display: none;
}

.lm-CommandPalette-content {
  background-color: var(--jp-layout-color1);
}

.lm-CommandPalette-content:empty::after {
  content: 'No results';
  margin: auto;
  margin-top: 20px;
  width: 100px;
  display: block;
  font-size: var(--jp-ui-font-size2);
  font-family: var(--jp-ui-font-family);
  font-weight: lighter;
}

.lm-CommandPalette-emptyMessage {
  text-align: center;
  margin-top: 24px;
  line-height: 1.32;
  padding: 0 8px;
  color: var(--jp-content-font-color3);
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Dialog {
  position: absolute;
  z-index: 10000;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  top: 0;
  left: 0;
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-dialog-background);
}

.jp-Dialog-content {
  display: flex;
  flex-direction: column;
  margin-left: auto;
  margin-right: auto;
  background: var(--jp-layout-color1);
  padding: 24px 24px 12px;
  min-width: 300px;
  min-height: 150px;
  max-width: 1000px;
  max-height: 500px;
  box-sizing: border-box;
  box-shadow: var(--jp-elevation-z20);
  word-wrap: break-word;
  border-radius: var(--jp-border-radius);

  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color1);
  resize: both;
}

.jp-Dialog-content.jp-Dialog-content-small {
  max-width: 500px;
}

.jp-Dialog-button {
  overflow: visible;
}

button.jp-Dialog-button:focus {
  outline: 1px solid var(--jp-brand-color1);
  outline-offset: 4px;
  -moz-outline-radius: 0;
}

button.jp-Dialog-button:focus::-moz-focus-inner {
  border: 0;
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-accept:focus,
button.jp-Dialog-button.jp-mod-styled.jp-mod-warn:focus,
button.jp-Dialog-button.jp-mod-styled.jp-mod-reject:focus {
  outline-offset: 4px;
  -moz-outline-radius: 0;
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-accept:focus {
  outline: 1px solid var(--jp-accept-color-normal, var(--jp-brand-color1));
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-warn:focus {
  outline: 1px solid var(--jp-warn-color-normal, var(--jp-error-color1));
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-reject:focus {
  outline: 1px solid var(--jp-reject-color-normal, var(--md-grey-600));
}

button.jp-Dialog-close-button {
  padding: 0;
  height: 100%;
  min-width: unset;
  min-height: unset;
}

.jp-Dialog-header {
  display: flex;
  justify-content: space-between;
  flex: 0 0 auto;
  padding-bottom: 12px;
  font-size: var(--jp-ui-font-size3);
  font-weight: 400;
  color: var(--jp-ui-font-color1);
}

.jp-Dialog-body {
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
  font-size: var(--jp-ui-font-size1);
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  overflow: auto;
}

.jp-Dialog-footer {
  display: flex;
  flex-direction: row;
  justify-content: flex-end;
  align-items: center;
  flex: 0 0 auto;
  margin-left: -12px;
  margin-right: -12px;
  padding: 12px;
}

.jp-Dialog-checkbox {
  padding-right: 5px;
}

.jp-Dialog-checkbox > input:focus-visible {
  outline: 1px solid var(--jp-input-active-border-color);
  outline-offset: 1px;
}

.jp-Dialog-spacer {
  flex: 1 1 auto;
}

.jp-Dialog-title {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.jp-Dialog-body > .jp-select-wrapper {
  width: 100%;
}

.jp-Dialog-body > button {
  padding: 0 16px;
}

.jp-Dialog-body > label {
  line-height: 1.4;
  color: var(--jp-ui-font-color0);
}

.jp-Dialog-button.jp-mod-styled:not(:last-child) {
  margin-right: 12px;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-Input-Boolean-Dialog {
  flex-direction: row-reverse;
  align-items: end;
  width: 100%;
}

.jp-Input-Boolean-Dialog > label {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MainAreaWidget > :focus {
  outline: none;
}

.jp-MainAreaWidget .jp-MainAreaWidget-error {
  padding: 6px;
}

.jp-MainAreaWidget .jp-MainAreaWidget-error > pre {
  width: auto;
  padding: 10px;
  background: var(--jp-error-color3);
  border: var(--jp-border-width) solid var(--jp-error-color1);
  border-radius: var(--jp-border-radius);
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  white-space: pre-wrap;
  word-wrap: break-word;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/**
 * google-material-color v1.2.6
 * https://github.com/danlevan/google-material-color
 */
:root {
  --md-red-50: #ffebee;
  --md-red-100: #ffcdd2;
  --md-red-200: #ef9a9a;
  --md-red-300: #e57373;
  --md-red-400: #ef5350;
  --md-red-500: #f44336;
  --md-red-600: #e53935;
  --md-red-700: #d32f2f;
  --md-red-800: #c62828;
  --md-red-900: #b71c1c;
  --md-red-A100: #ff8a80;
  --md-red-A200: #ff5252;
  --md-red-A400: #ff1744;
  --md-red-A700: #d50000;
  --md-pink-50: #fce4ec;
  --md-pink-100: #f8bbd0;
  --md-pink-200: #f48fb1;
  --md-pink-300: #f06292;
  --md-pink-400: #ec407a;
  --md-pink-500: #e91e63;
  --md-pink-600: #d81b60;
  --md-pink-700: #c2185b;
  --md-pink-800: #ad1457;
  --md-pink-900: #880e4f;
  --md-pink-A100: #ff80ab;
  --md-pink-A200: #ff4081;
  --md-pink-A400: #f50057;
  --md-pink-A700: #c51162;
  --md-purple-50: #f3e5f5;
  --md-purple-100: #e1bee7;
  --md-purple-200: #ce93d8;
  --md-purple-300: #ba68c8;
  --md-purple-400: #ab47bc;
  --md-purple-500: #9c27b0;
  --md-purple-600: #8e24aa;
  --md-purple-700: #7b1fa2;
  --md-purple-800: #6a1b9a;
  --md-purple-900: #4a148c;
  --md-purple-A100: #ea80fc;
  --md-purple-A200: #e040fb;
  --md-purple-A400: #d500f9;
  --md-purple-A700: #a0f;
  --md-deep-purple-50: #ede7f6;
  --md-deep-purple-100: #d1c4e9;
  --md-deep-purple-200: #b39ddb;
  --md-deep-purple-300: #9575cd;
  --md-deep-purple-400: #7e57c2;
  --md-deep-purple-500: #673ab7;
  --md-deep-purple-600: #5e35b1;
  --md-deep-purple-700: #512da8;
  --md-deep-purple-800: #4527a0;
  --md-deep-purple-900: #311b92;
  --md-deep-purple-A100: #b388ff;
  --md-deep-purple-A200: #7c4dff;
  --md-deep-purple-A400: #651fff;
  --md-deep-purple-A700: #6200ea;
  --md-indigo-50: #e8eaf6;
  --md-indigo-100: #c5cae9;
  --md-indigo-200: #9fa8da;
  --md-indigo-300: #7986cb;
  --md-indigo-400: #5c6bc0;
  --md-indigo-500: #3f51b5;
  --md-indigo-600: #3949ab;
  --md-indigo-700: #303f9f;
  --md-indigo-800: #283593;
  --md-indigo-900: #1a237e;
  --md-indigo-A100: #8c9eff;
  --md-indigo-A200: #536dfe;
  --md-indigo-A400: #3d5afe;
  --md-indigo-A700: #304ffe;
  --md-blue-50: #e3f2fd;
  --md-blue-100: #bbdefb;
  --md-blue-200: #90caf9;
  --md-blue-300: #64b5f6;
  --md-blue-400: #42a5f5;
  --md-blue-500: #2196f3;
  --md-blue-600: #1e88e5;
  --md-blue-700: #1976d2;
  --md-blue-800: #1565c0;
  --md-blue-900: #0d47a1;
  --md-blue-A100: #82b1ff;
  --md-blue-A200: #448aff;
  --md-blue-A400: #2979ff;
  --md-blue-A700: #2962ff;
  --md-light-blue-50: #e1f5fe;
  --md-light-blue-100: #b3e5fc;
  --md-light-blue-200: #81d4fa;
  --md-light-blue-300: #4fc3f7;
  --md-light-blue-400: #29b6f6;
  --md-light-blue-500: #03a9f4;
  --md-light-blue-600: #039be5;
  --md-light-blue-700: #0288d1;
  --md-light-blue-800: #0277bd;
  --md-light-blue-900: #01579b;
  --md-light-blue-A100: #80d8ff;
  --md-light-blue-A200: #40c4ff;
  --md-light-blue-A400: #00b0ff;
  --md-light-blue-A700: #0091ea;
  --md-cyan-50: #e0f7fa;
  --md-cyan-100: #b2ebf2;
  --md-cyan-200: #80deea;
  --md-cyan-300: #4dd0e1;
  --md-cyan-400: #26c6da;
  --md-cyan-500: #00bcd4;
  --md-cyan-600: #00acc1;
  --md-cyan-700: #0097a7;
  --md-cyan-800: #00838f;
  --md-cyan-900: #006064;
  --md-cyan-A100: #84ffff;
  --md-cyan-A200: #18ffff;
  --md-cyan-A400: #00e5ff;
  --md-cyan-A700: #00b8d4;
  --md-teal-50: #e0f2f1;
  --md-teal-100: #b2dfdb;
  --md-teal-200: #80cbc4;
  --md-teal-300: #4db6ac;
  --md-teal-400: #26a69a;
  --md-teal-500: #009688;
  --md-teal-600: #00897b;
  --md-teal-700: #00796b;
  --md-teal-800: #00695c;
  --md-teal-900: #004d40;
  --md-teal-A100: #a7ffeb;
  --md-teal-A200: #64ffda;
  --md-teal-A400: #1de9b6;
  --md-teal-A700: #00bfa5;
  --md-green-50: #e8f5e9;
  --md-green-100: #c8e6c9;
  --md-green-200: #a5d6a7;
  --md-green-300: #81c784;
  --md-green-400: #66bb6a;
  --md-green-500: #4caf50;
  --md-green-600: #43a047;
  --md-green-700: #388e3c;
  --md-green-800: #2e7d32;
  --md-green-900: #1b5e20;
  --md-green-A100: #b9f6ca;
  --md-green-A200: #69f0ae;
  --md-green-A400: #00e676;
  --md-green-A700: #00c853;
  --md-light-green-50: #f1f8e9;
  --md-light-green-100: #dcedc8;
  --md-light-green-200: #c5e1a5;
  --md-light-green-300: #aed581;
  --md-light-green-400: #9ccc65;
  --md-light-green-500: #8bc34a;
  --md-light-green-600: #7cb342;
  --md-light-green-700: #689f38;
  --md-light-green-800: #558b2f;
  --md-light-green-900: #33691e;
  --md-light-green-A100: #ccff90;
  --md-light-green-A200: #b2ff59;
  --md-light-green-A400: #76ff03;
  --md-light-green-A700: #64dd17;
  --md-lime-50: #f9fbe7;
  --md-lime-100: #f0f4c3;
  --md-lime-200: #e6ee9c;
  --md-lime-300: #dce775;
  --md-lime-400: #d4e157;
  --md-lime-500: #cddc39;
  --md-lime-600: #c0ca33;
  --md-lime-700: #afb42b;
  --md-lime-800: #9e9d24;
  --md-lime-900: #827717;
  --md-lime-A100: #f4ff81;
  --md-lime-A200: #eeff41;
  --md-lime-A400: #c6ff00;
  --md-lime-A700: #aeea00;
  --md-yellow-50: #fffde7;
  --md-yellow-100: #fff9c4;
  --md-yellow-200: #fff59d;
  --md-yellow-300: #fff176;
  --md-yellow-400: #ffee58;
  --md-yellow-500: #ffeb3b;
  --md-yellow-600: #fdd835;
  --md-yellow-700: #fbc02d;
  --md-yellow-800: #f9a825;
  --md-yellow-900: #f57f17;
  --md-yellow-A100: #ffff8d;
  --md-yellow-A200: #ff0;
  --md-yellow-A400: #ffea00;
  --md-yellow-A700: #ffd600;
  --md-amber-50: #fff8e1;
  --md-amber-100: #ffecb3;
  --md-amber-200: #ffe082;
  --md-amber-300: #ffd54f;
  --md-amber-400: #ffca28;
  --md-amber-500: #ffc107;
  --md-amber-600: #ffb300;
  --md-amber-700: #ffa000;
  --md-amber-800: #ff8f00;
  --md-amber-900: #ff6f00;
  --md-amber-A100: #ffe57f;
  --md-amber-A200: #ffd740;
  --md-amber-A400: #ffc400;
  --md-amber-A700: #ffab00;
  --md-orange-50: #fff3e0;
  --md-orange-100: #ffe0b2;
  --md-orange-200: #ffcc80;
  --md-orange-300: #ffb74d;
  --md-orange-400: #ffa726;
  --md-orange-500: #ff9800;
  --md-orange-600: #fb8c00;
  --md-orange-700: #f57c00;
  --md-orange-800: #ef6c00;
  --md-orange-900: #e65100;
  --md-orange-A100: #ffd180;
  --md-orange-A200: #ffab40;
  --md-orange-A400: #ff9100;
  --md-orange-A700: #ff6d00;
  --md-deep-orange-50: #fbe9e7;
  --md-deep-orange-100: #ffccbc;
  --md-deep-orange-200: #ffab91;
  --md-deep-orange-300: #ff8a65;
  --md-deep-orange-400: #ff7043;
  --md-deep-orange-500: #ff5722;
  --md-deep-orange-600: #f4511e;
  --md-deep-orange-700: #e64a19;
  --md-deep-orange-800: #d84315;
  --md-deep-orange-900: #bf360c;
  --md-deep-orange-A100: #ff9e80;
  --md-deep-orange-A200: #ff6e40;
  --md-deep-orange-A400: #ff3d00;
  --md-deep-orange-A700: #dd2c00;
  --md-brown-50: #efebe9;
  --md-brown-100: #d7ccc8;
  --md-brown-200: #bcaaa4;
  --md-brown-300: #a1887f;
  --md-brown-400: #8d6e63;
  --md-brown-500: #795548;
  --md-brown-600: #6d4c41;
  --md-brown-700: #5d4037;
  --md-brown-800: #4e342e;
  --md-brown-900: #3e2723;
  --md-grey-50: #fafafa;
  --md-grey-100: #f5f5f5;
  --md-grey-200: #eee;
  --md-grey-300: #e0e0e0;
  --md-grey-400: #bdbdbd;
  --md-grey-500: #9e9e9e;
  --md-grey-600: #757575;
  --md-grey-700: #616161;
  --md-grey-800: #424242;
  --md-grey-900: #212121;
  --md-blue-grey-50: #eceff1;
  --md-blue-grey-100: #cfd8dc;
  --md-blue-grey-200: #b0bec5;
  --md-blue-grey-300: #90a4ae;
  --md-blue-grey-400: #78909c;
  --md-blue-grey-500: #607d8b;
  --md-blue-grey-600: #546e7a;
  --md-blue-grey-700: #455a64;
  --md-blue-grey-800: #37474f;
  --md-blue-grey-900: #263238;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| RenderedText
|----------------------------------------------------------------------------*/

:root {
  /* This is the padding value to fill the gaps between lines containing spans with background color. */
  --jp-private-code-span-padding: calc(
    (var(--jp-code-line-height) - 1) * var(--jp-code-font-size) / 2
  );
}

.jp-RenderedText {
  text-align: left;
  padding-left: var(--jp-code-padding);
  line-height: var(--jp-code-line-height);
  font-family: var(--jp-code-font-family);
}

.jp-RenderedText pre,
.jp-RenderedJavaScript pre,
.jp-RenderedHTMLCommon pre {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-code-font-size);
  border: none;
  margin: 0;
  padding: 0;
}

.jp-RenderedText pre a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

.jp-RenderedText pre a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}

.jp-RenderedText pre a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* console foregrounds and backgrounds */
.jp-RenderedText pre .ansi-black-fg {
  color: #3e424d;
}

.jp-RenderedText pre .ansi-red-fg {
  color: #e75c58;
}

.jp-RenderedText pre .ansi-green-fg {
  color: #00a250;
}

.jp-RenderedText pre .ansi-yellow-fg {
  color: #ddb62b;
}

.jp-RenderedText pre .ansi-blue-fg {
  color: #208ffb;
}

.jp-RenderedText pre .ansi-magenta-fg {
  color: #d160c4;
}

.jp-RenderedText pre .ansi-cyan-fg {
  color: #60c6c8;
}

.jp-RenderedText pre .ansi-white-fg {
  color: #c5c1b4;
}

.jp-RenderedText pre .ansi-black-bg {
  background-color: #3e424d;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-red-bg {
  background-color: #e75c58;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-green-bg {
  background-color: #00a250;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-yellow-bg {
  background-color: #ddb62b;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-blue-bg {
  background-color: #208ffb;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-magenta-bg {
  background-color: #d160c4;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-cyan-bg {
  background-color: #60c6c8;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-white-bg {
  background-color: #c5c1b4;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-black-intense-fg {
  color: #282c36;
}

.jp-RenderedText pre .ansi-red-intense-fg {
  color: #b22b31;
}

.jp-RenderedText pre .ansi-green-intense-fg {
  color: #007427;
}

.jp-RenderedText pre .ansi-yellow-intense-fg {
  color: #b27d12;
}

.jp-RenderedText pre .ansi-blue-intense-fg {
  color: #0065ca;
}

.jp-RenderedText pre .ansi-magenta-intense-fg {
  color: #a03196;
}

.jp-RenderedText pre .ansi-cyan-intense-fg {
  color: #258f8f;
}

.jp-RenderedText pre .ansi-white-intense-fg {
  color: #a1a6b2;
}

.jp-RenderedText pre .ansi-black-intense-bg {
  background-color: #282c36;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-red-intense-bg {
  background-color: #b22b31;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-green-intense-bg {
  background-color: #007427;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-yellow-intense-bg {
  background-color: #b27d12;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-blue-intense-bg {
  background-color: #0065ca;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-magenta-intense-bg {
  background-color: #a03196;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-cyan-intense-bg {
  background-color: #258f8f;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-white-intense-bg {
  background-color: #a1a6b2;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-default-inverse-fg {
  color: var(--jp-ui-inverse-font-color0);
}

.jp-RenderedText pre .ansi-default-inverse-bg {
  background-color: var(--jp-inverse-layout-color0);
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-bold {
  font-weight: bold;
}

.jp-RenderedText pre .ansi-underline {
  text-decoration: underline;
}

.jp-RenderedText[data-mime-type='application/vnd.jupyter.stderr'] {
  background: var(--jp-rendermime-error-background);
  padding-top: var(--jp-code-padding);
}

/*-----------------------------------------------------------------------------
| RenderedLatex
|----------------------------------------------------------------------------*/

.jp-RenderedLatex {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);
}

/* Left-justify outputs.*/
.jp-OutputArea-output.jp-RenderedLatex {
  padding: var(--jp-code-padding);
  text-align: left;
}

/*-----------------------------------------------------------------------------
| RenderedHTML
|----------------------------------------------------------------------------*/

.jp-RenderedHTMLCommon {
  color: var(--jp-content-font-color1);
  font-family: var(--jp-content-font-family);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);

  /* Give a bit more R padding on Markdown text to keep line lengths reasonable */
  padding-right: 20px;
}

.jp-RenderedHTMLCommon em {
  font-style: italic;
}

.jp-RenderedHTMLCommon strong {
  font-weight: bold;
}

.jp-RenderedHTMLCommon u {
  text-decoration: underline;
}

.jp-RenderedHTMLCommon a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* Headings */

.jp-RenderedHTMLCommon h1,
.jp-RenderedHTMLCommon h2,
.jp-RenderedHTMLCommon h3,
.jp-RenderedHTMLCommon h4,
.jp-RenderedHTMLCommon h5,
.jp-RenderedHTMLCommon h6 {
  line-height: var(--jp-content-heading-line-height);
  font-weight: var(--jp-content-heading-font-weight);
  font-style: normal;
  margin: var(--jp-content-heading-margin-top) 0
    var(--jp-content-heading-margin-bottom) 0;
}

.jp-RenderedHTMLCommon h1:first-child,
.jp-RenderedHTMLCommon h2:first-child,
.jp-RenderedHTMLCommon h3:first-child,
.jp-RenderedHTMLCommon h4:first-child,
.jp-RenderedHTMLCommon h5:first-child,
.jp-RenderedHTMLCommon h6:first-child {
  margin-top: calc(0.5 * var(--jp-content-heading-margin-top));
}

.jp-RenderedHTMLCommon h1:last-child,
.jp-RenderedHTMLCommon h2:last-child,
.jp-RenderedHTMLCommon h3:last-child,
.jp-RenderedHTMLCommon h4:last-child,
.jp-RenderedHTMLCommon h5:last-child,
.jp-RenderedHTMLCommon h6:last-child {
  margin-bottom: calc(0.5 * var(--jp-content-heading-margin-bottom));
}

.jp-RenderedHTMLCommon h1 {
  font-size: var(--jp-content-font-size5);
}

.jp-RenderedHTMLCommon h2 {
  font-size: var(--jp-content-font-size4);
}

.jp-RenderedHTMLCommon h3 {
  font-size: var(--jp-content-font-size3);
}

.jp-RenderedHTMLCommon h4 {
  font-size: var(--jp-content-font-size2);
}

.jp-RenderedHTMLCommon h5 {
  font-size: var(--jp-content-font-size1);
}

.jp-RenderedHTMLCommon h6 {
  font-size: var(--jp-content-font-size0);
}

/* Lists */

/* stylelint-disable selector-max-type, selector-max-compound-selectors */

.jp-RenderedHTMLCommon ul:not(.list-inline),
.jp-RenderedHTMLCommon ol:not(.list-inline) {
  padding-left: 2em;
}

.jp-RenderedHTMLCommon ul {
  list-style: disc;
}

.jp-RenderedHTMLCommon ul ul {
  list-style: square;
}

.jp-RenderedHTMLCommon ul ul ul {
  list-style: circle;
}

.jp-RenderedHTMLCommon ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol ol {
  list-style: upper-alpha;
}

.jp-RenderedHTMLCommon ol ol ol {
  list-style: lower-alpha;
}

.jp-RenderedHTMLCommon ol ol ol ol {
  list-style: lower-roman;
}

.jp-RenderedHTMLCommon ol ol ol ol ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol,
.jp-RenderedHTMLCommon ul {
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon ul ul,
.jp-RenderedHTMLCommon ul ol,
.jp-RenderedHTMLCommon ol ul,
.jp-RenderedHTMLCommon ol ol {
  margin-bottom: 0;
}

/* stylelint-enable selector-max-type, selector-max-compound-selectors */

.jp-RenderedHTMLCommon hr {
  color: var(--jp-border-color2);
  background-color: var(--jp-border-color1);
  margin-top: 1em;
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon > pre {
  margin: 1.5em 2em;
}

.jp-RenderedHTMLCommon pre,
.jp-RenderedHTMLCommon code {
  border: 0;
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  line-height: var(--jp-code-line-height);
  padding: 0;
  white-space: pre-wrap;
}

.jp-RenderedHTMLCommon :not(pre) > code {
  background-color: var(--jp-layout-color2);
  padding: 1px 5px;
}

/* Tables */

.jp-RenderedHTMLCommon table {
  border-collapse: collapse;
  border-spacing: 0;
  border: none;
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  table-layout: fixed;
  margin-left: auto;
  margin-bottom: 1em;
  margin-right: auto;
}

.jp-RenderedHTMLCommon thead {
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  vertical-align: bottom;
}

.jp-RenderedHTMLCommon td,
.jp-RenderedHTMLCommon th,
.jp-RenderedHTMLCommon tr {
  vertical-align: middle;
  padding: 0.5em;
  line-height: normal;
  white-space: normal;
  max-width: none;
  border: none;
}

.jp-RenderedMarkdown.jp-RenderedHTMLCommon td,
.jp-RenderedMarkdown.jp-RenderedHTMLCommon th {
  max-width: none;
}

:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon td,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon th,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon tr {
  text-align: right;
}

.jp-RenderedHTMLCommon th {
  font-weight: bold;
}

.jp-RenderedHTMLCommon tbody tr:nth-child(odd) {
  background: var(--jp-layout-color0);
}

.jp-RenderedHTMLCommon tbody tr:nth-child(even) {
  background: var(--jp-rendermime-table-row-background);
}

.jp-RenderedHTMLCommon tbody tr:hover {
  background: var(--jp-rendermime-table-row-hover-background);
}

.jp-RenderedHTMLCommon p {
  text-align: left;
  margin: 0;
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon img {
  -moz-force-broken-image-icon: 1;
}

/* Restrict to direct children as other images could be nested in other content. */
.jp-RenderedHTMLCommon > img {
  display: block;
  margin-left: 0;
  margin-right: 0;
  margin-bottom: 1em;
}

/* Change color behind transparent images if they need it... */
[data-jp-theme-light='false'] .jp-RenderedImage img.jp-needs-light-background {
  background-color: var(--jp-inverse-layout-color1);
}

[data-jp-theme-light='true'] .jp-RenderedImage img.jp-needs-dark-background {
  background-color: var(--jp-inverse-layout-color1);
}

.jp-RenderedHTMLCommon img,
.jp-RenderedImage img,
.jp-RenderedHTMLCommon svg,
.jp-RenderedSVG svg {
  max-width: 100%;
  height: auto;
}

.jp-RenderedHTMLCommon img.jp-mod-unconfined,
.jp-RenderedImage img.jp-mod-unconfined,
.jp-RenderedHTMLCommon svg.jp-mod-unconfined,
.jp-RenderedSVG svg.jp-mod-unconfined {
  max-width: none;
}

.jp-RenderedHTMLCommon .alert {
  padding: var(--jp-notebook-padding);
  border: var(--jp-border-width) solid transparent;
  border-radius: var(--jp-border-radius);
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon .alert-info {
  color: var(--jp-info-color0);
  background-color: var(--jp-info-color3);
  border-color: var(--jp-info-color2);
}

.jp-RenderedHTMLCommon .alert-info hr {
  border-color: var(--jp-info-color3);
}

.jp-RenderedHTMLCommon .alert-info > p:last-child,
.jp-RenderedHTMLCommon .alert-info > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-warning {
  color: var(--jp-warn-color0);
  background-color: var(--jp-warn-color3);
  border-color: var(--jp-warn-color2);
}

.jp-RenderedHTMLCommon .alert-warning hr {
  border-color: var(--jp-warn-color3);
}

.jp-RenderedHTMLCommon .alert-warning > p:last-child,
.jp-RenderedHTMLCommon .alert-warning > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-success {
  color: var(--jp-success-color0);
  background-color: var(--jp-success-color3);
  border-color: var(--jp-success-color2);
}

.jp-RenderedHTMLCommon .alert-success hr {
  border-color: var(--jp-success-color3);
}

.jp-RenderedHTMLCommon .alert-success > p:last-child,
.jp-RenderedHTMLCommon .alert-success > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-danger {
  color: var(--jp-error-color0);
  background-color: var(--jp-error-color3);
  border-color: var(--jp-error-color2);
}

.jp-RenderedHTMLCommon .alert-danger hr {
  border-color: var(--jp-error-color3);
}

.jp-RenderedHTMLCommon .alert-danger > p:last-child,
.jp-RenderedHTMLCommon .alert-danger > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon blockquote {
  margin: 1em 2em;
  padding: 0 1em;
  border-left: 5px solid var(--jp-border-color2);
}

a.jp-InternalAnchorLink {
  visibility: hidden;
  margin-left: 8px;
  color: var(--md-blue-800);
}

h1:hover .jp-InternalAnchorLink,
h2:hover .jp-InternalAnchorLink,
h3:hover .jp-InternalAnchorLink,
h4:hover .jp-InternalAnchorLink,
h5:hover .jp-InternalAnchorLink,
h6:hover .jp-InternalAnchorLink {
  visibility: visible;
}

.jp-RenderedHTMLCommon kbd {
  background-color: var(--jp-rendermime-table-row-background);
  border: 1px solid var(--jp-border-color0);
  border-bottom-color: var(--jp-border-color2);
  border-radius: 3px;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
  display: inline-block;
  font-size: var(--jp-ui-font-size0);
  line-height: 1em;
  padding: 0.2em 0.5em;
}

/* Most direct children of .jp-RenderedHTMLCommon have a margin-bottom of 1.0.
 * At the bottom of cells this is a bit too much as there is also spacing
 * between cells. Going all the way to 0 gets too tight between markdown and
 * code cells.
 */
.jp-RenderedHTMLCommon > *:last-child {
  margin-bottom: 0.5em;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-cursor-backdrop {
  position: fixed;
  width: 200px;
  height: 200px;
  margin-top: -100px;
  margin-left: -100px;
  will-change: transform;
  z-index: 100;
}

.lm-mod-drag-image {
  will-change: transform;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-lineFormSearch {
  padding: 4px 12px;
  background-color: var(--jp-layout-color2);
  box-shadow: var(--jp-toolbar-box-shadow);
  z-index: 2;
  font-size: var(--jp-ui-font-size1);
}

.jp-lineFormCaption {
  font-size: var(--jp-ui-font-size0);
  line-height: var(--jp-ui-font-size1);
  margin-top: 4px;
  color: var(--jp-ui-font-color0);
}

.jp-baseLineForm {
  border: none;
  border-radius: 0;
  position: absolute;
  background-size: 16px;
  background-repeat: no-repeat;
  background-position: center;
  outline: none;
}

.jp-lineFormButtonContainer {
  top: 4px;
  right: 8px;
  height: 24px;
  padding: 0 12px;
  width: 12px;
}

.jp-lineFormButtonIcon {
  top: 0;
  right: 0;
  background-color: var(--jp-brand-color1);
  height: 100%;
  width: 100%;
  box-sizing: border-box;
  padding: 4px 6px;
}

.jp-lineFormButton {
  top: 0;
  right: 0;
  background-color: transparent;
  height: 100%;
  width: 100%;
  box-sizing: border-box;
}

.jp-lineFormWrapper {
  overflow: hidden;
  padding: 0 8px;
  border: 1px solid var(--jp-border-color0);
  background-color: var(--jp-input-active-background);
  height: 22px;
}

.jp-lineFormWrapperFocusWithin {
  border: var(--jp-border-width) solid var(--md-blue-500);
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-lineFormInput {
  background: transparent;
  width: 200px;
  height: 100%;
  border: none;
  outline: none;
  color: var(--jp-ui-font-color0);
  line-height: 28px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-JSONEditor {
  display: flex;
  flex-direction: column;
  width: 100%;
}

.jp-JSONEditor-host {
  flex: 1 1 auto;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0;
  background: var(--jp-layout-color0);
  min-height: 50px;
  padding: 1px;
}

.jp-JSONEditor.jp-mod-error .jp-JSONEditor-host {
  border-color: red;
  outline-color: red;
}

.jp-JSONEditor-header {
  display: flex;
  flex: 1 0 auto;
  padding: 0 0 0 12px;
}

.jp-JSONEditor-header label {
  flex: 0 0 auto;
}

.jp-JSONEditor-commitButton {
  height: 16px;
  width: 16px;
  background-size: 18px;
  background-repeat: no-repeat;
  background-position: center;
}

.jp-JSONEditor-host.jp-mod-focused {
  background-color: var(--jp-input-active-background);
  border: 1px solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}

.jp-Editor.jp-mod-dropTarget {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/
.jp-DocumentSearch-input {
  border: none;
  outline: none;
  color: var(--jp-ui-font-color0);
  font-size: var(--jp-ui-font-size1);
  background-color: var(--jp-layout-color0);
  font-family: var(--jp-ui-font-family);
  padding: 2px 1px;
  resize: none;
}

.jp-DocumentSearch-overlay {
  position: absolute;
  background-color: var(--jp-toolbar-background);
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  border-left: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  top: 0;
  right: 0;
  z-index: 7;
  min-width: 405px;
  padding: 2px;
  font-size: var(--jp-ui-font-size1);

  --jp-private-document-search-button-height: 20px;
}

.jp-DocumentSearch-overlay button {
  background-color: var(--jp-toolbar-background);
  outline: 0;
}

.jp-DocumentSearch-overlay button:hover {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-overlay button:active {
  background-color: var(--jp-layout-color3);
}

.jp-DocumentSearch-overlay-row {
  display: flex;
  align-items: center;
  margin-bottom: 2px;
}

.jp-DocumentSearch-button-content {
  display: inline-block;
  cursor: pointer;
  box-sizing: border-box;
  width: 100%;
  height: 100%;
}

.jp-DocumentSearch-button-content svg {
  width: 100%;
  height: 100%;
}

.jp-DocumentSearch-input-wrapper {
  border: var(--jp-border-width) solid var(--jp-border-color0);
  display: flex;
  background-color: var(--jp-layout-color0);
  margin: 2px;
}

.jp-DocumentSearch-input-wrapper:focus-within {
  border-color: var(--jp-cell-editor-active-border-color);
}

.jp-DocumentSearch-toggle-wrapper,
.jp-DocumentSearch-button-wrapper {
  all: initial;
  overflow: hidden;
  display: inline-block;
  border: none;
  box-sizing: border-box;
}

.jp-DocumentSearch-toggle-wrapper {
  width: 14px;
  height: 14px;
}

.jp-DocumentSearch-button-wrapper {
  width: var(--jp-private-document-search-button-height);
  height: var(--jp-private-document-search-button-height);
}

.jp-DocumentSearch-toggle-wrapper:focus,
.jp-DocumentSearch-button-wrapper:focus {
  outline: var(--jp-border-width) solid
    var(--jp-cell-editor-active-border-color);
  outline-offset: -1px;
}

.jp-DocumentSearch-toggle-wrapper,
.jp-DocumentSearch-button-wrapper,
.jp-DocumentSearch-button-content:focus {
  outline: none;
}

.jp-DocumentSearch-toggle-placeholder {
  width: 5px;
}

.jp-DocumentSearch-input-button::before {
  display: block;
  padding-top: 100%;
}

.jp-DocumentSearch-input-button-off {
  opacity: var(--jp-search-toggle-off-opacity);
}

.jp-DocumentSearch-input-button-off:hover {
  opacity: var(--jp-search-toggle-hover-opacity);
}

.jp-DocumentSearch-input-button-on {
  opacity: var(--jp-search-toggle-on-opacity);
}

.jp-DocumentSearch-index-counter {
  padding-left: 10px;
  padding-right: 10px;
  user-select: none;
  min-width: 35px;
  display: inline-block;
}

.jp-DocumentSearch-up-down-wrapper {
  display: inline-block;
  padding-right: 2px;
  margin-left: auto;
  white-space: nowrap;
}

.jp-DocumentSearch-spacer {
  margin-left: auto;
}

.jp-DocumentSearch-up-down-wrapper button {
  outline: 0;
  border: none;
  width: var(--jp-private-document-search-button-height);
  height: var(--jp-private-document-search-button-height);
  vertical-align: middle;
  margin: 1px 5px 2px;
}

.jp-DocumentSearch-up-down-button:hover {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-up-down-button:active {
  background-color: var(--jp-layout-color3);
}

.jp-DocumentSearch-filter-button {
  border-radius: var(--jp-border-radius);
}

.jp-DocumentSearch-filter-button:hover {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-filter-button-enabled {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-filter-button-enabled:hover {
  background-color: var(--jp-layout-color3);
}

.jp-DocumentSearch-search-options {
  padding: 0 8px;
  margin-left: 3px;
  width: 100%;
  display: grid;
  justify-content: start;
  grid-template-columns: 1fr 1fr;
  align-items: center;
  justify-items: stretch;
}

.jp-DocumentSearch-search-filter-disabled {
  color: var(--jp-ui-font-color2);
}

.jp-DocumentSearch-search-filter {
  display: flex;
  align-items: center;
  user-select: none;
}

.jp-DocumentSearch-regex-error {
  color: var(--jp-error-color0);
}

.jp-DocumentSearch-replace-button-wrapper {
  overflow: hidden;
  display: inline-block;
  box-sizing: border-box;
  border: var(--jp-border-width) solid var(--jp-border-color0);
  margin: auto 2px;
  padding: 1px 4px;
  height: calc(var(--jp-private-document-search-button-height) + 2px);
}

.jp-DocumentSearch-replace-button-wrapper:focus {
  border: var(--jp-border-width) solid var(--jp-cell-editor-active-border-color);
}

.jp-DocumentSearch-replace-button {
  display: inline-block;
  text-align: center;
  cursor: pointer;
  box-sizing: border-box;
  color: var(--jp-ui-font-color1);

  /* height - 2 * (padding of wrapper) */
  line-height: calc(var(--jp-private-document-search-button-height) - 2px);
  width: 100%;
  height: 100%;
}

.jp-DocumentSearch-replace-button:focus {
  outline: none;
}

.jp-DocumentSearch-replace-wrapper-class {
  margin-left: 14px;
  display: flex;
}

.jp-DocumentSearch-replace-toggle {
  border: none;
  background-color: var(--jp-toolbar-background);
  border-radius: var(--jp-border-radius);
}

.jp-DocumentSearch-replace-toggle:hover {
  background-color: var(--jp-layout-color2);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.cm-editor {
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  border: 0;
  border-radius: 0;
  height: auto;

  /* Changed to auto to autogrow */
}

.cm-editor pre {
  padding: 0 var(--jp-code-padding);
}

.jp-CodeMirrorEditor[data-type='inline'] .cm-dialog {
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
}

.jp-CodeMirrorEditor {
  cursor: text;
}

/* When zoomed out 67% and 33% on a screen of 1440 width x 900 height */
@media screen and (min-width: 2138px) and (max-width: 4319px) {
  .jp-CodeMirrorEditor[data-type='inline'] .cm-cursor {
    border-left: var(--jp-code-cursor-width1) solid
      var(--jp-editor-cursor-color);
  }
}

/* When zoomed out less than 33% */
@media screen and (min-width: 4320px) {
  .jp-CodeMirrorEditor[data-type='inline'] .cm-cursor {
    border-left: var(--jp-code-cursor-width2) solid
      var(--jp-editor-cursor-color);
  }
}

.cm-editor.jp-mod-readOnly .cm-cursor {
  display: none;
}

.jp-CollaboratorCursor {
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
  border-top: none;
  border-bottom: 3px solid;
  background-clip: content-box;
  margin-left: -5px;
  margin-right: -5px;
}

.cm-searching,
.cm-searching span {
  /* `.cm-searching span`: we need to override syntax highlighting */
  background-color: var(--jp-search-unselected-match-background-color);
  color: var(--jp-search-unselected-match-color);
}

.cm-searching::selection,
.cm-searching span::selection {
  background-color: var(--jp-search-unselected-match-background-color);
  color: var(--jp-search-unselected-match-color);
}

.jp-current-match > .cm-searching,
.jp-current-match > .cm-searching span,
.cm-searching > .jp-current-match,
.cm-searching > .jp-current-match span {
  background-color: var(--jp-search-selected-match-background-color);
  color: var(--jp-search-selected-match-color);
}

.jp-current-match > .cm-searching::selection,
.cm-searching > .jp-current-match::selection,
.jp-current-match > .cm-searching span::selection {
  background-color: var(--jp-search-selected-match-background-color);
  color: var(--jp-search-selected-match-color);
}

.cm-trailingspace {
  background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAFCAYAAAB4ka1VAAAAsElEQVQIHQGlAFr/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA7+r3zKmT0/+pk9P/7+r3zAAAAAAAAAAABAAAAAAAAAAA6OPzM+/q9wAAAAAA6OPzMwAAAAAAAAAAAgAAAAAAAAAAGR8NiRQaCgAZIA0AGR8NiQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQyoYJ/SY80UAAAAASUVORK5CYII=);
  background-position: center left;
  background-repeat: repeat-x;
}

.jp-CollaboratorCursor-hover {
  position: absolute;
  z-index: 1;
  transform: translateX(-50%);
  color: white;
  border-radius: 3px;
  padding-left: 4px;
  padding-right: 4px;
  padding-top: 1px;
  padding-bottom: 1px;
  text-align: center;
  font-size: var(--jp-ui-font-size1);
  white-space: nowrap;
}

.jp-CodeMirror-ruler {
  border-left: 1px dashed var(--jp-border-color2);
}

/* Styles for shared cursors (remote cursor locations and selected ranges) */
.jp-CodeMirrorEditor .cm-ySelectionCaret {
  position: relative;
  border-left: 1px solid black;
  margin-left: -1px;
  margin-right: -1px;
  box-sizing: border-box;
}

.jp-CodeMirrorEditor .cm-ySelectionCaret > .cm-ySelectionInfo {
  white-space: nowrap;
  position: absolute;
  top: -1.15em;
  padding-bottom: 0.05em;
  left: -1px;
  font-size: 0.95em;
  font-family: var(--jp-ui-font-family);
  font-weight: bold;
  line-height: normal;
  user-select: none;
  color: white;
  padding-left: 2px;
  padding-right: 2px;
  z-index: 101;
  transition: opacity 0.3s ease-in-out;
}

.jp-CodeMirrorEditor .cm-ySelectionInfo {
  transition-delay: 0.7s;
  opacity: 0;
}

.jp-CodeMirrorEditor .cm-ySelectionCaret:hover > .cm-ySelectionInfo {
  opacity: 1;
  transition-delay: 0s;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MimeDocument {
  outline: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-filebrowser-button-height: 28px;
  --jp-private-filebrowser-button-width: 48px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-FileBrowser .jp-SidePanel-content {
  display: flex;
  flex-direction: column;
}

.jp-FileBrowser-toolbar.jp-Toolbar {
  flex-wrap: wrap;
  row-gap: 12px;
  border-bottom: none;
  height: auto;
  margin: 8px 12px 0;
  box-shadow: none;
  padding: 0;
  justify-content: flex-start;
}

.jp-FileBrowser-Panel {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
}

.jp-BreadCrumbs {
  flex: 0 0 auto;
  margin: 8px 12px;
}

.jp-BreadCrumbs-item {
  margin: 0 2px;
  padding: 0 2px;
  border-radius: var(--jp-border-radius);
  cursor: pointer;
}

.jp-BreadCrumbs-item:hover {
  background-color: var(--jp-layout-color2);
}

.jp-BreadCrumbs-item:first-child {
  margin-left: 0;
}

.jp-BreadCrumbs-item.jp-mod-dropTarget {
  background-color: var(--jp-brand-color2);
  opacity: 0.7;
}

/*-----------------------------------------------------------------------------
| Buttons
|----------------------------------------------------------------------------*/

.jp-FileBrowser-toolbar > .jp-Toolbar-item {
  flex: 0 0 auto;
  padding-left: 0;
  padding-right: 2px;
  align-items: center;
  height: unset;
}

.jp-FileBrowser-toolbar > .jp-Toolbar-item .jp-ToolbarButtonComponent {
  width: 40px;
}

/*-----------------------------------------------------------------------------
| Other styles
|----------------------------------------------------------------------------*/

.jp-FileDialog.jp-mod-conflict input {
  color: var(--jp-error-color1);
}

.jp-FileDialog .jp-new-name-title {
  margin-top: 12px;
}

.jp-LastModified-hidden {
  display: none;
}

.jp-FileSize-hidden {
  display: none;
}

.jp-FileBrowser .lm-AccordionPanel > h3:first-child {
  display: none;
}

/*-----------------------------------------------------------------------------
| DirListing
|----------------------------------------------------------------------------*/

.jp-DirListing {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  outline: 0;
}

.jp-DirListing-header {
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  align-items: center;
  overflow: hidden;
  border-top: var(--jp-border-width) solid var(--jp-border-color2);
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  box-shadow: var(--jp-toolbar-box-shadow);
  z-index: 2;
}

.jp-DirListing-headerItem {
  padding: 4px 12px 2px;
  font-weight: 500;
}

.jp-DirListing-headerItem:hover {
  background: var(--jp-layout-color2);
}

.jp-DirListing-headerItem.jp-id-name {
  flex: 1 0 84px;
}

.jp-DirListing-headerItem.jp-id-modified {
  flex: 0 0 112px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
}

.jp-DirListing-headerItem.jp-id-filesize {
  flex: 0 0 75px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
}

.jp-id-narrow {
  display: none;
  flex: 0 0 5px;
  padding: 4px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
  color: var(--jp-border-color2);
}

.jp-DirListing-narrow .jp-id-narrow {
  display: block;
}

.jp-DirListing-narrow .jp-id-modified,
.jp-DirListing-narrow .jp-DirListing-itemModified {
  display: none;
}

.jp-DirListing-headerItem.jp-mod-selected {
  font-weight: 600;
}

/* increase specificity to override bundled default */
.jp-DirListing-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  list-style-type: none;
  overflow: auto;
  background-color: var(--jp-layout-color1);
}

.jp-DirListing-content mark {
  color: var(--jp-ui-font-color0);
  background-color: transparent;
  font-weight: bold;
}

.jp-DirListing-content .jp-DirListing-item.jp-mod-selected mark {
  color: var(--jp-ui-inverse-font-color0);
}

/* Style the directory listing content when a user drops a file to upload */
.jp-DirListing.jp-mod-native-drop .jp-DirListing-content {
  outline: 5px dashed rgba(128, 128, 128, 0.5);
  outline-offset: -10px;
  cursor: copy;
}

.jp-DirListing-item {
  display: flex;
  flex-direction: row;
  align-items: center;
  padding: 4px 12px;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-DirListing-checkboxWrapper {
  /* Increases hit area of checkbox. */
  padding: 4px;
}

.jp-DirListing-header
  .jp-DirListing-checkboxWrapper
  + .jp-DirListing-headerItem {
  padding-left: 4px;
}

.jp-DirListing-content .jp-DirListing-checkboxWrapper {
  position: relative;
  left: -4px;
  margin: -4px 0 -4px -8px;
}

.jp-DirListing-checkboxWrapper.jp-mod-visible {
  visibility: visible;
}

/* For devices that support hovering, hide checkboxes until hovered, selected...
*/
@media (hover: hover) {
  .jp-DirListing-checkboxWrapper {
    visibility: hidden;
  }

  .jp-DirListing-item:hover .jp-DirListing-checkboxWrapper,
  .jp-DirListing-item.jp-mod-selected .jp-DirListing-checkboxWrapper {
    visibility: visible;
  }
}

.jp-DirListing-item[data-is-dot] {
  opacity: 75%;
}

.jp-DirListing-item.jp-mod-selected {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.jp-DirListing-item.jp-mod-dropTarget {
  background: var(--jp-brand-color3);
}

.jp-DirListing-item:hover:not(.jp-mod-selected) {
  background: var(--jp-layout-color2);
}

.jp-DirListing-itemIcon {
  flex: 0 0 20px;
  margin-right: 4px;
}

.jp-DirListing-itemText {
  flex: 1 0 64px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  user-select: none;
}

.jp-DirListing-itemText:focus {
  outline-width: 2px;
  outline-color: var(--jp-inverse-layout-color1);
  outline-style: solid;
  outline-offset: 1px;
}

.jp-DirListing-item.jp-mod-selected .jp-DirListing-itemText:focus {
  outline-color: var(--jp-layout-color1);
}

.jp-DirListing-itemModified {
  flex: 0 0 125px;
  text-align: right;
}

.jp-DirListing-itemFileSize {
  flex: 0 0 90px;
  text-align: right;
}

.jp-DirListing-editor {
  flex: 1 0 64px;
  outline: none;
  border: none;
  color: var(--jp-ui-font-color1);
  background-color: var(--jp-layout-color1);
}

.jp-DirListing-item.jp-mod-running .jp-DirListing-itemIcon::before {
  color: var(--jp-success-color1);
  content: '\25CF';
  font-size: 8px;
  position: absolute;
  left: -8px;
}

.jp-DirListing-item.jp-mod-running.jp-mod-selected
  .jp-DirListing-itemIcon::before {
  color: var(--jp-ui-inverse-font-color1);
}

.jp-DirListing-item.lm-mod-drag-image,
.jp-DirListing-item.jp-mod-selected.lm-mod-drag-image {
  font-size: var(--jp-ui-font-size1);
  padding-left: 4px;
  margin-left: 4px;
  width: 160px;
  background-color: var(--jp-ui-inverse-font-color2);
  box-shadow: var(--jp-elevation-z2);
  border-radius: 0;
  color: var(--jp-ui-font-color1);
  transform: translateX(-40%) translateY(-58%);
}

.jp-Document {
  min-width: 120px;
  min-height: 120px;
  outline: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Main OutputArea
| OutputArea has a list of Outputs
|----------------------------------------------------------------------------*/

.jp-OutputArea {
  overflow-y: auto;
}

.jp-OutputArea-child {
  display: table;
  table-layout: fixed;
  width: 100%;
  overflow: hidden;
}

.jp-OutputPrompt {
  width: var(--jp-cell-prompt-width);
  color: var(--jp-cell-outprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
  opacity: var(--jp-cell-prompt-opacity);

  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;

  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-OutputArea-prompt {
  display: table-cell;
  vertical-align: top;
}

.jp-OutputArea-output {
  display: table-cell;
  width: 100%;
  height: auto;
  overflow: auto;
  user-select: text;
  -moz-user-select: text;
  -webkit-user-select: text;
  -ms-user-select: text;
}

.jp-OutputArea .jp-RenderedText {
  padding-left: 1ch;
}

/**
 * Prompt overlay.
 */

.jp-OutputArea-promptOverlay {
  position: absolute;
  top: 0;
  width: var(--jp-cell-prompt-width);
  height: 100%;
  opacity: 0.5;
}

.jp-OutputArea-promptOverlay:hover {
  background: var(--jp-layout-color2);
  box-shadow: inset 0 0 1px var(--jp-inverse-layout-color0);
  cursor: zoom-out;
}

.jp-mod-outputsScrolled .jp-OutputArea-promptOverlay:hover {
  cursor: zoom-in;
}

/**
 * Isolated output.
 */
.jp-OutputArea-output.jp-mod-isolated {
  width: 100%;
  display: block;
}

/*
When drag events occur, `lm-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated {
  position: relative;
}

body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}

/* pre */

.jp-OutputArea-output pre {
  border: none;
  margin: 0;
  padding: 0;
  overflow-x: auto;
  overflow-y: auto;
  word-break: break-all;
  word-wrap: break-word;
  white-space: pre-wrap;
}

/* tables */

.jp-OutputArea-output.jp-RenderedHTMLCommon table {
  margin-left: 0;
  margin-right: 0;
}

/* description lists */

.jp-OutputArea-output dl,
.jp-OutputArea-output dt,
.jp-OutputArea-output dd {
  display: block;
}

.jp-OutputArea-output dl {
  width: 100%;
  overflow: hidden;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dt {
  font-weight: bold;
  float: left;
  width: 20%;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dd {
  float: left;
  width: 80%;
  padding: 0;
  margin: 0;
}

.jp-TrimmedOutputs pre {
  background: var(--jp-layout-color3);
  font-size: calc(var(--jp-code-font-size) * 1.4);
  text-align: center;
  text-transform: uppercase;
}

/* Hide the gutter in case of
 *  - nested output areas (e.g. in the case of output widgets)
 *  - mirrored output areas
 */
.jp-OutputArea .jp-OutputArea .jp-OutputArea-prompt {
  display: none;
}

/* Hide empty lines in the output area, for instance due to cleared widgets */
.jp-OutputArea-prompt:empty {
  padding: 0;
  border: 0;
}

/*-----------------------------------------------------------------------------
| executeResult is added to any Output-result for the display of the object
| returned by a cell
|----------------------------------------------------------------------------*/

.jp-OutputArea-output.jp-OutputArea-executeResult {
  margin-left: 0;
  width: 100%;
}

/* Text output with the Out[] prompt needs a top padding to match the
 * alignment of the Out[] prompt itself.
 */
.jp-OutputArea-executeResult .jp-RenderedText.jp-OutputArea-output {
  padding-top: var(--jp-code-padding);
  border-top: var(--jp-border-width) solid transparent;
}

/*-----------------------------------------------------------------------------
| The Stdin output
|----------------------------------------------------------------------------*/

.jp-Stdin-prompt {
  color: var(--jp-content-font-color0);
  padding-right: var(--jp-code-padding);
  vertical-align: baseline;
  flex: 0 0 auto;
}

.jp-Stdin-input {
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  color: inherit;
  background-color: inherit;
  width: 42%;
  min-width: 200px;

  /* make sure input baseline aligns with prompt */
  vertical-align: baseline;

  /* padding + margin = 0.5em between prompt and cursor */
  padding: 0 0.25em;
  margin: 0 0.25em;
  flex: 0 0 70%;
}

.jp-Stdin-input::placeholder {
  opacity: 0;
}

.jp-Stdin-input:focus {
  box-shadow: none;
}

.jp-Stdin-input:focus::placeholder {
  opacity: 1;
}

/*-----------------------------------------------------------------------------
| Output Area View
|----------------------------------------------------------------------------*/

.jp-LinkedOutputView .jp-OutputArea {
  height: 100%;
  display: block;
}

.jp-LinkedOutputView .jp-OutputArea-output:only-child {
  height: 100%;
}

/*-----------------------------------------------------------------------------
| Printing
|----------------------------------------------------------------------------*/

@media print {
  .jp-OutputArea-child {
    break-inside: avoid-page;
  }
}

/*-----------------------------------------------------------------------------
| Mobile
|----------------------------------------------------------------------------*/
@media only screen and (max-width: 760px) {
  .jp-OutputPrompt {
    display: table-row;
    text-align: left;
  }

  .jp-OutputArea-child .jp-OutputArea-output {
    display: table-row;
    margin-left: var(--jp-notebook-padding);
  }
}

/* Trimmed outputs warning */
.jp-TrimmedOutputs > a {
  margin: 10px;
  text-decoration: none;
  cursor: pointer;
}

.jp-TrimmedOutputs > a:hover {
  text-decoration: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Table of Contents
|----------------------------------------------------------------------------*/

:root {
  --jp-private-toc-active-width: 4px;
}

.jp-TableOfContents {
  display: flex;
  flex-direction: column;
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  height: 100%;
}

.jp-TableOfContents-placeholder {
  text-align: center;
}

.jp-TableOfContents-placeholderContent {
  color: var(--jp-content-font-color2);
  padding: 8px;
}

.jp-TableOfContents-placeholderContent > h3 {
  margin-bottom: var(--jp-content-heading-margin-bottom);
}

.jp-TableOfContents .jp-SidePanel-content {
  overflow-y: auto;
}

.jp-TableOfContents-tree {
  margin: 4px;
}

.jp-TableOfContents ol {
  list-style-type: none;
}

/* stylelint-disable-next-line selector-max-type */
.jp-TableOfContents li > ol {
  /* Align left border with triangle icon center */
  padding-left: 11px;
}

.jp-TableOfContents-content {
  /* left margin for the active heading indicator */
  margin: 0 0 0 var(--jp-private-toc-active-width);
  padding: 0;
  background-color: var(--jp-layout-color1);
}

.jp-tocItem {
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-tocItem-heading {
  display: flex;
  cursor: pointer;
}

.jp-tocItem-heading:hover {
  background-color: var(--jp-layout-color2);
}

.jp-tocItem-content {
  display: block;
  padding: 4px 0;
  white-space: nowrap;
  text-overflow: ellipsis;
  overflow-x: hidden;
}

.jp-tocItem-collapser {
  height: 20px;
  margin: 2px 2px 0;
  padding: 0;
  background: none;
  border: none;
  cursor: pointer;
}

.jp-tocItem-collapser:hover {
  background-color: var(--jp-layout-color3);
}

/* Active heading indicator */

.jp-tocItem-heading::before {
  content: ' ';
  background: transparent;
  width: var(--jp-private-toc-active-width);
  height: 24px;
  position: absolute;
  left: 0;
  border-radius: var(--jp-border-radius);
}

.jp-tocItem-heading.jp-tocItem-active::before {
  background-color: var(--jp-brand-color1);
}

.jp-tocItem-heading:hover.jp-tocItem-active::before {
  background: var(--jp-brand-color0);
  opacity: 1;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapser {
  flex: 0 0 var(--jp-cell-collapser-width);
  padding: 0;
  margin: 0;
  border: none;
  outline: none;
  background: transparent;
  border-radius: var(--jp-border-radius);
  opacity: 1;
}

.jp-Collapser-child {
  display: block;
  width: 100%;
  box-sizing: border-box;

  /* height: 100% doesn't work because the height of its parent is computed from content */
  position: absolute;
  top: 0;
  bottom: 0;
}

/*-----------------------------------------------------------------------------
| Printing
|----------------------------------------------------------------------------*/

/*
Hiding collapsers in print mode.

Note: input and output wrappers have "display: block" propery in print mode.
*/

@media print {
  .jp-Collapser {
    display: none;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Header/Footer
|----------------------------------------------------------------------------*/

/* Hidden by zero height by default */
.jp-CellHeader,
.jp-CellFooter {
  height: 0;
  width: 100%;
  padding: 0;
  margin: 0;
  border: none;
  outline: none;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Input
|----------------------------------------------------------------------------*/

/* All input areas */
.jp-InputArea {
  display: table;
  table-layout: fixed;
  width: 100%;
  overflow: hidden;
}

.jp-InputArea-editor {
  display: table-cell;
  overflow: hidden;
  vertical-align: top;

  /* This is the non-active, default styling */
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  border-radius: 0;
  background: var(--jp-cell-editor-background);
}

.jp-InputPrompt {
  display: table-cell;
  vertical-align: top;
  width: var(--jp-cell-prompt-width);
  color: var(--jp-cell-inprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  opacity: var(--jp-cell-prompt-opacity);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;

  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;

  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

/*-----------------------------------------------------------------------------
| Mobile
|----------------------------------------------------------------------------*/
@media only screen and (max-width: 760px) {
  .jp-InputArea-editor {
    display: table-row;
    margin-left: var(--jp-notebook-padding);
  }

  .jp-InputPrompt {
    display: table-row;
    text-align: left;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Placeholder
|----------------------------------------------------------------------------*/

.jp-Placeholder {
  display: table;
  table-layout: fixed;
  width: 100%;
}

.jp-Placeholder-prompt {
  display: table-cell;
  box-sizing: border-box;
}

.jp-Placeholder-content {
  display: table-cell;
  padding: 4px 6px;
  border: 1px solid transparent;
  border-radius: 0;
  background: none;
  box-sizing: border-box;
  cursor: pointer;
}

.jp-Placeholder-contentContainer {
  display: flex;
}

.jp-Placeholder-content:hover,
.jp-InputPlaceholder > .jp-Placeholder-content:hover {
  border-color: var(--jp-layout-color3);
}

.jp-Placeholder-content .jp-MoreHorizIcon {
  width: 32px;
  height: 16px;
  border: 1px solid transparent;
  border-radius: var(--jp-border-radius);
}

.jp-Placeholder-content .jp-MoreHorizIcon:hover {
  border: 1px solid var(--jp-border-color1);
  box-shadow: 0 0 2px 0 rgba(0, 0, 0, 0.25);
  background-color: var(--jp-layout-color0);
}

.jp-PlaceholderText {
  white-space: nowrap;
  overflow-x: hidden;
  color: var(--jp-inverse-layout-color3);
  font-family: var(--jp-code-font-family);
}

.jp-InputPlaceholder > .jp-Placeholder-content {
  border-color: var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Private CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-cell-scrolling-output-offset: 5px;
}

/*-----------------------------------------------------------------------------
| Cell
|----------------------------------------------------------------------------*/

.jp-Cell {
  padding: var(--jp-cell-padding);
  margin: 0;
  border: none;
  outline: none;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Common input/output
|----------------------------------------------------------------------------*/

.jp-Cell-inputWrapper,
.jp-Cell-outputWrapper {
  display: flex;
  flex-direction: row;
  padding: 0;
  margin: 0;

  /* Added to reveal the box-shadow on the input and output collapsers. */
  overflow: visible;
}

/* Only input/output areas inside cells */
.jp-Cell-inputArea,
.jp-Cell-outputArea {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Collapser
|----------------------------------------------------------------------------*/

/* Make the output collapser disappear when there is not output, but do so
 * in a manner that leaves it in the layout and preserves its width.
 */
.jp-Cell.jp-mod-noOutputs .jp-Cell-outputCollapser {
  border: none !important;
  background: transparent !important;
}

.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputCollapser {
  min-height: var(--jp-cell-collapser-min-height);
}

/*-----------------------------------------------------------------------------
| Output
|----------------------------------------------------------------------------*/

/* Put a space between input and output when there IS output */
.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputWrapper {
  margin-top: 5px;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea {
  overflow-y: auto;
  max-height: 24em;
  margin-left: var(--jp-private-cell-scrolling-output-offset);
  resize: vertical;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea[style*='height'] {
  max-height: unset;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea::after {
  content: ' ';
  box-shadow: inset 0 0 6px 2px rgb(0 0 0 / 30%);
  width: 100%;
  height: 100%;
  position: sticky;
  bottom: 0;
  top: 0;
  margin-top: -50%;
  float: left;
  display: block;
  pointer-events: none;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-child {
  padding-top: 6px;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-prompt {
  width: calc(
    var(--jp-cell-prompt-width) - var(--jp-private-cell-scrolling-output-offset)
  );
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-promptOverlay {
  left: calc(-1 * var(--jp-private-cell-scrolling-output-offset));
}

/*-----------------------------------------------------------------------------
| CodeCell
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| MarkdownCell
|----------------------------------------------------------------------------*/

.jp-MarkdownOutput {
  display: table-cell;
  width: 100%;
  margin-top: 0;
  margin-bottom: 0;
  padding-left: var(--jp-code-padding);
}

.jp-MarkdownOutput.jp-RenderedHTMLCommon {
  overflow: auto;
}

/* collapseHeadingButton (show always if hiddenCellsButton is _not_ shown) */
.jp-collapseHeadingButton {
  display: flex;
  min-height: var(--jp-cell-collapser-min-height);
  font-size: var(--jp-code-font-size);
  position: absolute;
  background-color: transparent;
  background-size: 25px;
  background-repeat: no-repeat;
  background-position-x: center;
  background-position-y: top;
  background-image: var(--jp-icon-caret-down);
  right: 0;
  top: 0;
  bottom: 0;
}

.jp-collapseHeadingButton.jp-mod-collapsed {
  background-image: var(--jp-icon-caret-right);
}

/*
 set the container font size to match that of content
 so that the nested collapse buttons have the right size
*/
.jp-MarkdownCell .jp-InputPrompt {
  font-size: var(--jp-content-font-size1);
}

/*
  Align collapseHeadingButton with cell top header
  The font sizes are identical to the ones in packages/rendermime/style/base.css
*/
.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='1'] {
  font-size: var(--jp-content-font-size5);
  background-position-y: calc(0.3 * var(--jp-content-font-size5));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='2'] {
  font-size: var(--jp-content-font-size4);
  background-position-y: calc(0.3 * var(--jp-content-font-size4));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='3'] {
  font-size: var(--jp-content-font-size3);
  background-position-y: calc(0.3 * var(--jp-content-font-size3));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='4'] {
  font-size: var(--jp-content-font-size2);
  background-position-y: calc(0.3 * var(--jp-content-font-size2));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='5'] {
  font-size: var(--jp-content-font-size1);
  background-position-y: top;
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='6'] {
  font-size: var(--jp-content-font-size0);
  background-position-y: top;
}

/* collapseHeadingButton (show only on (hover,active) if hiddenCellsButton is shown) */
.jp-Notebook.jp-mod-showHiddenCellsButton .jp-collapseHeadingButton {
  display: none;
}

.jp-Notebook.jp-mod-showHiddenCellsButton
  :is(.jp-MarkdownCell:hover, .jp-mod-active)
  .jp-collapseHeadingButton {
  display: flex;
}

/* showHiddenCellsButton (only show if jp-mod-showHiddenCellsButton is set, which
is a consequence of the showHiddenCellsButton option in Notebook Settings)*/
.jp-Notebook.jp-mod-showHiddenCellsButton .jp-showHiddenCellsButton {
  margin-left: calc(var(--jp-cell-prompt-width) + 2 * var(--jp-code-padding));
  margin-top: var(--jp-code-padding);
  border: 1px solid var(--jp-border-color2);
  background-color: var(--jp-border-color3) !important;
  color: var(--jp-content-font-color0) !important;
  display: flex;
}

.jp-Notebook.jp-mod-showHiddenCellsButton .jp-showHiddenCellsButton:hover {
  background-color: var(--jp-border-color2) !important;
}

.jp-showHiddenCellsButton {
  display: none;
}

/*-----------------------------------------------------------------------------
| Printing
|----------------------------------------------------------------------------*/

/*
Using block instead of flex to allow the use of the break-inside CSS property for
cell outputs.
*/

@media print {
  .jp-Cell-inputWrapper,
  .jp-Cell-outputWrapper {
    display: block;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-notebook-toolbar-padding: 2px 5px 2px 2px;
}

/*-----------------------------------------------------------------------------

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-NotebookPanel-toolbar {
  padding: var(--jp-notebook-toolbar-padding);

  /* disable paint containment from lumino 2.0 default strict CSS containment */
  contain: style size !important;
}

.jp-Toolbar-item.jp-Notebook-toolbarCellType .jp-select-wrapper.jp-mod-focused {
  border: none;
  box-shadow: none;
}

.jp-Notebook-toolbarCellTypeDropdown select {
  height: 24px;
  font-size: var(--jp-ui-font-size1);
  line-height: 14px;
  border-radius: 0;
  display: block;
}

.jp-Notebook-toolbarCellTypeDropdown span {
  top: 5px !important;
}

.jp-Toolbar-responsive-popup {
  position: absolute;
  height: fit-content;
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  justify-content: flex-end;
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  background: var(--jp-toolbar-background);
  min-height: var(--jp-toolbar-micro-height);
  padding: var(--jp-notebook-toolbar-padding);
  z-index: 1;
  right: 0;
  top: 0;
}

.jp-Toolbar > .jp-Toolbar-responsive-opener {
  margin-left: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-Notebook-ExecutionIndicator {
  position: relative;
  display: inline-block;
  height: 100%;
  z-index: 9997;
}

.jp-Notebook-ExecutionIndicator-tooltip {
  visibility: hidden;
  height: auto;
  width: max-content;
  width: -moz-max-content;
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color1);
  text-align: justify;
  border-radius: 6px;
  padding: 0 5px;
  position: fixed;
  display: table;
}

.jp-Notebook-ExecutionIndicator-tooltip.up {
  transform: translateX(-50%) translateY(-100%) translateY(-32px);
}

.jp-Notebook-ExecutionIndicator-tooltip.down {
  transform: translateX(calc(-100% + 16px)) translateY(5px);
}

.jp-Notebook-ExecutionIndicator-tooltip.hidden {
  display: none;
}

.jp-Notebook-ExecutionIndicator:hover .jp-Notebook-ExecutionIndicator-tooltip {
  visibility: visible;
}

.jp-Notebook-ExecutionIndicator span {
  font-size: var(--jp-ui-font-size1);
  font-family: var(--jp-ui-font-family);
  color: var(--jp-ui-font-color1);
  line-height: 24px;
  display: block;
}

.jp-Notebook-ExecutionIndicator-progress-bar {
  display: flex;
  justify-content: center;
  height: 100%;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*
 * Execution indicator
 */
.jp-tocItem-content::after {
  content: '';

  /* Must be identical to form a circle */
  width: 12px;
  height: 12px;
  background: none;
  border: none;
  position: absolute;
  right: 0;
}

.jp-tocItem-content[data-running='0']::after {
  border-radius: 50%;
  border: var(--jp-border-width) solid var(--jp-inverse-layout-color3);
  background: none;
}

.jp-tocItem-content[data-running='1']::after {
  border-radius: 50%;
  border: var(--jp-border-width) solid var(--jp-inverse-layout-color3);
  background-color: var(--jp-inverse-layout-color3);
}

.jp-tocItem-content[data-running='0'],
.jp-tocItem-content[data-running='1'] {
  margin-right: 12px;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-Notebook-footer {
  height: 27px;
  margin-left: calc(
    var(--jp-cell-prompt-width) + var(--jp-cell-collapser-width) +
      var(--jp-cell-padding)
  );
  width: calc(
    100% -
      (
        var(--jp-cell-prompt-width) + var(--jp-cell-collapser-width) +
          var(--jp-cell-padding) + var(--jp-cell-padding)
      )
  );
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  color: var(--jp-ui-font-color3);
  margin-top: 6px;
  background: none;
  cursor: pointer;
}

.jp-Notebook-footer:focus {
  border-color: var(--jp-cell-editor-active-border-color);
}

/* For devices that support hovering, hide footer until hover */
@media (hover: hover) {
  .jp-Notebook-footer {
    opacity: 0;
  }

  .jp-Notebook-footer:focus,
  .jp-Notebook-footer:hover {
    opacity: 1;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Imports
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-side-by-side-output-size: 1fr;
  --jp-side-by-side-resized-cell: var(--jp-side-by-side-output-size);
  --jp-private-notebook-dragImage-width: 304px;
  --jp-private-notebook-dragImage-height: 36px;
  --jp-private-notebook-selected-color: var(--md-blue-400);
  --jp-private-notebook-active-color: var(--md-green-400);
}

/*-----------------------------------------------------------------------------
| Notebook
|----------------------------------------------------------------------------*/

/* stylelint-disable selector-max-class */

.jp-NotebookPanel {
  display: block;
  height: 100%;
}

.jp-NotebookPanel.jp-Document {
  min-width: 240px;
  min-height: 120px;
}

.jp-Notebook {
  padding: var(--jp-notebook-padding);
  outline: none;
  overflow: auto;
  background: var(--jp-layout-color0);
}

.jp-Notebook.jp-mod-scrollPastEnd::after {
  display: block;
  content: '';
  min-height: var(--jp-notebook-scroll-padding);
}

.jp-MainAreaWidget-ContainStrict .jp-Notebook * {
  contain: strict;
}

.jp-Notebook .jp-Cell {
  overflow: visible;
}

.jp-Notebook .jp-Cell .jp-InputPrompt {
  cursor: move;
}

/*-----------------------------------------------------------------------------
| Notebook state related styling
|
| The notebook and cells each have states, here are the possibilities:
|
| - Notebook
|   - Command
|   - Edit
| - Cell
|   - None
|   - Active (only one can be active)
|   - Selected (the cells actions are applied to)
|   - Multiselected (when multiple selected, the cursor)
|   - No outputs
|----------------------------------------------------------------------------*/

/* Command or edit modes */

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-InputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-OutputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

/* cell is active */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser {
  background: var(--jp-brand-color1);
}

/* cell is dirty */
.jp-Notebook .jp-Cell.jp-mod-dirty .jp-InputPrompt {
  color: var(--jp-warn-color1);
}

.jp-Notebook .jp-Cell.jp-mod-dirty .jp-InputPrompt::before {
  color: var(--jp-warn-color1);
  content: '•';
}

.jp-Notebook .jp-Cell.jp-mod-active.jp-mod-dirty .jp-Collapser {
  background: var(--jp-warn-color1);
}

/* collapser is hovered */
.jp-Notebook .jp-Cell .jp-Collapser:hover {
  box-shadow: var(--jp-elevation-z2);
  background: var(--jp-brand-color1);
  opacity: var(--jp-cell-collapser-not-active-hover-opacity);
}

/* cell is active and collapser is hovered */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser:hover {
  background: var(--jp-brand-color0);
  opacity: 1;
}

/* Command mode */

.jp-Notebook.jp-mod-commandMode .jp-Cell.jp-mod-selected {
  background: var(--jp-notebook-multiselected-color);
}

.jp-Notebook.jp-mod-commandMode
  .jp-Cell.jp-mod-active.jp-mod-selected:not(.jp-mod-multiSelected) {
  background: transparent;
}

/* Edit mode */

.jp-Notebook.jp-mod-editMode .jp-Cell.jp-mod-active .jp-InputArea-editor {
  border: var(--jp-border-width) solid var(--jp-cell-editor-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-cell-editor-active-background);
}

/*-----------------------------------------------------------------------------
| Notebook drag and drop
|----------------------------------------------------------------------------*/

.jp-Notebook-cell.jp-mod-dropSource {
  opacity: 0.5;
}

.jp-Notebook-cell.jp-mod-dropTarget,
.jp-Notebook.jp-mod-commandMode
  .jp-Notebook-cell.jp-mod-active.jp-mod-selected.jp-mod-dropTarget {
  border-top-color: var(--jp-private-notebook-selected-color);
  border-top-style: solid;
  border-top-width: 2px;
}

.jp-dragImage {
  display: block;
  flex-direction: row;
  width: var(--jp-private-notebook-dragImage-width);
  height: var(--jp-private-notebook-dragImage-height);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background);
  overflow: visible;
}

.jp-dragImage-singlePrompt {
  box-shadow: 2px 2px 4px 0 rgba(0, 0, 0, 0.12);
}

.jp-dragImage .jp-dragImage-content {
  flex: 1 1 auto;
  z-index: 2;
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  line-height: var(--jp-code-line-height);
  padding: var(--jp-code-padding);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background-color);
  color: var(--jp-content-font-color3);
  text-align: left;
  margin: 4px 4px 4px 0;
}

.jp-dragImage .jp-dragImage-prompt {
  flex: 0 0 auto;
  min-width: 36px;
  color: var(--jp-cell-inprompt-font-color);
  padding: var(--jp-code-padding);
  padding-left: 12px;
  font-family: var(--jp-cell-prompt-font-family);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: 1.9;
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
}

.jp-dragImage-multipleBack {
  z-index: -1;
  position: absolute;
  height: 32px;
  width: 300px;
  top: 8px;
  left: 8px;
  background: var(--jp-layout-color2);
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  box-shadow: 2px 2px 4px 0 rgba(0, 0, 0, 0.12);
}

/*-----------------------------------------------------------------------------
| Cell toolbar
|----------------------------------------------------------------------------*/

.jp-NotebookTools {
  display: block;
  min-width: var(--jp-sidebar-min-width);
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);

  /* This is needed so that all font sizing of children done in ems is
    * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  overflow: auto;
}

.jp-ActiveCellTool {
  padding: 12px 0;
  display: flex;
}

.jp-ActiveCellTool-Content {
  flex: 1 1 auto;
}

.jp-ActiveCellTool .jp-ActiveCellTool-CellContent {
  background: var(--jp-cell-editor-background);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  border-radius: 0;
  min-height: 29px;
}

.jp-ActiveCellTool .jp-InputPrompt {
  min-width: calc(var(--jp-cell-prompt-width) * 0.75);
}

.jp-ActiveCellTool-CellContent > pre {
  padding: 5px 4px;
  margin: 0;
  white-space: normal;
}

.jp-MetadataEditorTool {
  flex-direction: column;
  padding: 12px 0;
}

.jp-RankedPanel > :not(:first-child) {
  margin-top: 12px;
}

.jp-KeySelector select.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: var(--jp-border-width) solid var(--jp-border-color1);
}

.jp-KeySelector label,
.jp-MetadataEditorTool label,
.jp-NumberSetter label {
  line-height: 1.4;
}

.jp-NotebookTools .jp-select-wrapper {
  margin-top: 4px;
  margin-bottom: 0;
}

.jp-NumberSetter input {
  width: 100%;
  margin-top: 4px;
}

.jp-NotebookTools .jp-Collapse {
  margin-top: 16px;
}

/*-----------------------------------------------------------------------------
| Presentation Mode (.jp-mod-presentationMode)
|----------------------------------------------------------------------------*/

.jp-mod-presentationMode .jp-Notebook {
  --jp-content-font-size1: var(--jp-content-presentation-font-size1);
  --jp-code-font-size: var(--jp-code-presentation-font-size);
}

.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-InputPrompt,
.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-OutputPrompt {
  flex: 0 0 110px;
}

/*-----------------------------------------------------------------------------
| Side-by-side Mode (.jp-mod-sideBySide)
|----------------------------------------------------------------------------*/
.jp-mod-sideBySide.jp-Notebook .jp-Notebook-cell {
  margin-top: 3em;
  margin-bottom: 3em;
  margin-left: 5%;
  margin-right: 5%;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell {
  display: grid;
  grid-template-columns: minmax(0, 1fr) min-content minmax(
      0,
      var(--jp-side-by-side-output-size)
    );
  grid-template-rows: auto minmax(0, 1fr) auto;
  grid-template-areas:
    'header header header'
    'input handle output'
    'footer footer footer';
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell.jp-mod-resizedCell {
  grid-template-columns: minmax(0, 1fr) min-content minmax(
      0,
      var(--jp-side-by-side-resized-cell)
    );
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellHeader {
  grid-area: header;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-Cell-inputWrapper {
  grid-area: input;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-Cell-outputWrapper {
  /* overwrite the default margin (no vertical separation needed in side by side move */
  margin-top: 0;
  grid-area: output;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellFooter {
  grid-area: footer;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellResizeHandle {
  grid-area: handle;
  user-select: none;
  display: block;
  height: 100%;
  cursor: ew-resize;
  padding: 0 var(--jp-cell-padding);
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellResizeHandle::after {
  content: '';
  display: block;
  background: var(--jp-border-color2);
  height: 100%;
  width: 5px;
}

.jp-mod-sideBySide.jp-Notebook
  .jp-CodeCell.jp-mod-resizedCell
  .jp-CellResizeHandle::after {
  background: var(--jp-border-color0);
}

.jp-CellResizeHandle {
  display: none;
}

/*-----------------------------------------------------------------------------
| Placeholder
|----------------------------------------------------------------------------*/

.jp-Cell-Placeholder {
  padding-left: 55px;
}

.jp-Cell-Placeholder-wrapper {
  background: #fff;
  border: 1px solid;
  border-color: #e5e6e9 #dfe0e4 #d0d1d5;
  border-radius: 4px;
  -webkit-border-radius: 4px;
  margin: 10px 15px;
}

.jp-Cell-Placeholder-wrapper-inner {
  padding: 15px;
  position: relative;
}

.jp-Cell-Placeholder-wrapper-body {
  background-repeat: repeat;
  background-size: 50% auto;
}

.jp-Cell-Placeholder-wrapper-body div {
  background: #f6f7f8;
  background-image: -webkit-linear-gradient(
    left,
    #f6f7f8 0%,
    #edeef1 20%,
    #f6f7f8 40%,
    #f6f7f8 100%
  );
  background-repeat: no-repeat;
  background-size: 800px 104px;
  height: 104px;
  position: absolute;
  right: 15px;
  left: 15px;
  top: 15px;
}

div.jp-Cell-Placeholder-h1 {
  top: 20px;
  height: 20px;
  left: 15px;
  width: 150px;
}

div.jp-Cell-Placeholder-h2 {
  left: 15px;
  top: 50px;
  height: 10px;
  width: 100px;
}

div.jp-Cell-Placeholder-content-1,
div.jp-Cell-Placeholder-content-2,
div.jp-Cell-Placeholder-content-3 {
  left: 15px;
  right: 15px;
  height: 10px;
}

div.jp-Cell-Placeholder-content-1 {
  top: 100px;
}

div.jp-Cell-Placeholder-content-2 {
  top: 120px;
}

div.jp-Cell-Placeholder-content-3 {
  top: 140px;
}

</style>
<style type="text/css">
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
The following CSS variables define the main, public API for styling JupyterLab.
These variables should be used by all plugins wherever possible. In other
words, plugins should not define custom colors, sizes, etc unless absolutely
necessary. This enables users to change the visual theme of JupyterLab
by changing these variables.

Many variables appear in an ordered sequence (0,1,2,3). These sequences
are designed to work well together, so for example, `--jp-border-color1` should
be used with `--jp-layout-color1`. The numbers have the following meanings:

* 0: super-primary, reserved for special emphasis
* 1: primary, most important under normal situations
* 2: secondary, next most important under normal situations
* 3: tertiary, next most important under normal situations

Throughout JupyterLab, we are mostly following principles from Google's
Material Design when selecting colors. We are not, however, following
all of MD as it is not optimized for dense, information rich UIs.
*/

:root {
  /* Elevation
   *
   * We style box-shadows using Material Design's idea of elevation. These particular numbers are taken from here:
   *
   * https://github.com/material-components/material-components-web
   * https://material-components-web.appspot.com/elevation.html
   */

  --jp-shadow-base-lightness: 0;
  --jp-shadow-umbra-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.2
  );
  --jp-shadow-penumbra-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.14
  );
  --jp-shadow-ambient-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.12
  );
  --jp-elevation-z0: none;
  --jp-elevation-z1: 0 2px 1px -1px var(--jp-shadow-umbra-color),
    0 1px 1px 0 var(--jp-shadow-penumbra-color),
    0 1px 3px 0 var(--jp-shadow-ambient-color);
  --jp-elevation-z2: 0 3px 1px -2px var(--jp-shadow-umbra-color),
    0 2px 2px 0 var(--jp-shadow-penumbra-color),
    0 1px 5px 0 var(--jp-shadow-ambient-color);
  --jp-elevation-z4: 0 2px 4px -1px var(--jp-shadow-umbra-color),
    0 4px 5px 0 var(--jp-shadow-penumbra-color),
    0 1px 10px 0 var(--jp-shadow-ambient-color);
  --jp-elevation-z6: 0 3px 5px -1px var(--jp-shadow-umbra-color),
    0 6px 10px 0 var(--jp-shadow-penumbra-color),
    0 1px 18px 0 var(--jp-shadow-ambient-color);
  --jp-elevation-z8: 0 5px 5px -3px var(--jp-shadow-umbra-color),
    0 8px 10px 1px var(--jp-shadow-penumbra-color),
    0 3px 14px 2px var(--jp-shadow-ambient-color);
  --jp-elevation-z12: 0 7px 8px -4px var(--jp-shadow-umbra-color),
    0 12px 17px 2px var(--jp-shadow-penumbra-color),
    0 5px 22px 4px var(--jp-shadow-ambient-color);
  --jp-elevation-z16: 0 8px 10px -5px var(--jp-shadow-umbra-color),
    0 16px 24px 2px var(--jp-shadow-penumbra-color),
    0 6px 30px 5px var(--jp-shadow-ambient-color);
  --jp-elevation-z20: 0 10px 13px -6px var(--jp-shadow-umbra-color),
    0 20px 31px 3px var(--jp-shadow-penumbra-color),
    0 8px 38px 7px var(--jp-shadow-ambient-color);
  --jp-elevation-z24: 0 11px 15px -7px var(--jp-shadow-umbra-color),
    0 24px 38px 3px var(--jp-shadow-penumbra-color),
    0 9px 46px 8px var(--jp-shadow-ambient-color);

  /* Borders
   *
   * The following variables, specify the visual styling of borders in JupyterLab.
   */

  --jp-border-width: 1px;
  --jp-border-color0: var(--md-grey-400);
  --jp-border-color1: var(--md-grey-400);
  --jp-border-color2: var(--md-grey-300);
  --jp-border-color3: var(--md-grey-200);
  --jp-inverse-border-color: var(--md-grey-600);
  --jp-border-radius: 2px;

  /* UI Fonts
   *
   * The UI font CSS variables are used for the typography all of the JupyterLab
   * user interface elements that are not directly user generated content.
   *
   * The font sizing here is done assuming that the body font size of --jp-ui-font-size1
   * is applied to a parent element. When children elements, such as headings, are sized
   * in em all things will be computed relative to that body size.
   */

  --jp-ui-font-scale-factor: 1.2;
  --jp-ui-font-size0: 0.83333em;
  --jp-ui-font-size1: 13px; /* Base font size */
  --jp-ui-font-size2: 1.2em;
  --jp-ui-font-size3: 1.44em;
  --jp-ui-font-family: system-ui, -apple-system, blinkmacsystemfont, 'Segoe UI',
    helvetica, arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji',
    'Segoe UI Symbol';

  /*
   * Use these font colors against the corresponding main layout colors.
   * In a light theme, these go from dark to light.
   */

  /* Defaults use Material Design specification */
  --jp-ui-font-color0: rgba(0, 0, 0, 1);
  --jp-ui-font-color1: rgba(0, 0, 0, 0.87);
  --jp-ui-font-color2: rgba(0, 0, 0, 0.54);
  --jp-ui-font-color3: rgba(0, 0, 0, 0.38);

  /*
   * Use these against the brand/accent/warn/error colors.
   * These will typically go from light to darker, in both a dark and light theme.
   */

  --jp-ui-inverse-font-color0: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color1: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color2: rgba(255, 255, 255, 0.7);
  --jp-ui-inverse-font-color3: rgba(255, 255, 255, 0.5);

  /* Content Fonts
   *
   * Content font variables are used for typography of user generated content.
   *
   * The font sizing here is done assuming that the body font size of --jp-content-font-size1
   * is applied to a parent element. When children elements, such as headings, are sized
   * in em all things will be computed relative to that body size.
   */

  --jp-content-line-height: 1.6;
  --jp-content-font-scale-factor: 1.2;
  --jp-content-font-size0: 0.83333em;
  --jp-content-font-size1: 14px; /* Base font size */
  --jp-content-font-size2: 1.2em;
  --jp-content-font-size3: 1.44em;
  --jp-content-font-size4: 1.728em;
  --jp-content-font-size5: 2.0736em;

  /* This gives a magnification of about 125% in presentation mode over normal. */
  --jp-content-presentation-font-size1: 17px;
  --jp-content-heading-line-height: 1;
  --jp-content-heading-margin-top: 1.2em;
  --jp-content-heading-margin-bottom: 0.8em;
  --jp-content-heading-font-weight: 500;

  /* Defaults use Material Design specification */
  --jp-content-font-color0: rgba(0, 0, 0, 1);
  --jp-content-font-color1: rgba(0, 0, 0, 0.87);
  --jp-content-font-color2: rgba(0, 0, 0, 0.54);
  --jp-content-font-color3: rgba(0, 0, 0, 0.38);
  --jp-content-link-color: var(--md-blue-900);
  --jp-content-font-family: system-ui, -apple-system, blinkmacsystemfont,
    'Segoe UI', helvetica, arial, sans-serif, 'Apple Color Emoji',
    'Segoe UI Emoji', 'Segoe UI Symbol';

  /*
   * Code Fonts
   *
   * Code font variables are used for typography of code and other monospaces content.
   */

  --jp-code-font-size: 13px;
  --jp-code-line-height: 1.3077; /* 17px for 13px base */
  --jp-code-padding: 5px; /* 5px for 13px base, codemirror highlighting needs integer px value */
  --jp-code-font-family-default: menlo, consolas, 'DejaVu Sans Mono', monospace;
  --jp-code-font-family: var(--jp-code-font-family-default);

  /* This gives a magnification of about 125% in presentation mode over normal. */
  --jp-code-presentation-font-size: 16px;

  /* may need to tweak cursor width if you change font size */
  --jp-code-cursor-width0: 1.4px;
  --jp-code-cursor-width1: 2px;
  --jp-code-cursor-width2: 4px;

  /* Layout
   *
   * The following are the main layout colors use in JupyterLab. In a light
   * theme these would go from light to dark.
   */

  --jp-layout-color0: white;
  --jp-layout-color1: white;
  --jp-layout-color2: var(--md-grey-200);
  --jp-layout-color3: var(--md-grey-400);
  --jp-layout-color4: var(--md-grey-600);

  /* Inverse Layout
   *
   * The following are the inverse layout colors use in JupyterLab. In a light
   * theme these would go from dark to light.
   */

  --jp-inverse-layout-color0: #111;
  --jp-inverse-layout-color1: var(--md-grey-900);
  --jp-inverse-layout-color2: var(--md-grey-800);
  --jp-inverse-layout-color3: var(--md-grey-700);
  --jp-inverse-layout-color4: var(--md-grey-600);

  /* Brand/accent */

  --jp-brand-color0: var(--md-blue-900);
  --jp-brand-color1: var(--md-blue-700);
  --jp-brand-color2: var(--md-blue-300);
  --jp-brand-color3: var(--md-blue-100);
  --jp-brand-color4: var(--md-blue-50);
  --jp-accent-color0: var(--md-green-900);
  --jp-accent-color1: var(--md-green-700);
  --jp-accent-color2: var(--md-green-300);
  --jp-accent-color3: var(--md-green-100);

  /* State colors (warn, error, success, info) */

  --jp-warn-color0: var(--md-orange-900);
  --jp-warn-color1: var(--md-orange-700);
  --jp-warn-color2: var(--md-orange-300);
  --jp-warn-color3: var(--md-orange-100);
  --jp-error-color0: var(--md-red-900);
  --jp-error-color1: var(--md-red-700);
  --jp-error-color2: var(--md-red-300);
  --jp-error-color3: var(--md-red-100);
  --jp-success-color0: var(--md-green-900);
  --jp-success-color1: var(--md-green-700);
  --jp-success-color2: var(--md-green-300);
  --jp-success-color3: var(--md-green-100);
  --jp-info-color0: var(--md-cyan-900);
  --jp-info-color1: var(--md-cyan-700);
  --jp-info-color2: var(--md-cyan-300);
  --jp-info-color3: var(--md-cyan-100);

  /* Cell specific styles */

  --jp-cell-padding: 5px;
  --jp-cell-collapser-width: 8px;
  --jp-cell-collapser-min-height: 20px;
  --jp-cell-collapser-not-active-hover-opacity: 0.6;
  --jp-cell-editor-background: var(--md-grey-100);
  --jp-cell-editor-border-color: var(--md-grey-300);
  --jp-cell-editor-box-shadow: inset 0 0 2px var(--md-blue-300);
  --jp-cell-editor-active-background: var(--jp-layout-color0);
  --jp-cell-editor-active-border-color: var(--jp-brand-color1);
  --jp-cell-prompt-width: 64px;
  --jp-cell-prompt-font-family: var(--jp-code-font-family-default);
  --jp-cell-prompt-letter-spacing: 0;
  --jp-cell-prompt-opacity: 1;
  --jp-cell-prompt-not-active-opacity: 0.5;
  --jp-cell-prompt-not-active-font-color: var(--md-grey-700);

  /* A custom blend of MD grey and blue 600
   * See https://meyerweb.com/eric/tools/color-blend/#546E7A:1E88E5:5:hex */
  --jp-cell-inprompt-font-color: #307fc1;

  /* A custom blend of MD grey and orange 600
   * https://meyerweb.com/eric/tools/color-blend/#546E7A:F4511E:5:hex */
  --jp-cell-outprompt-font-color: #bf5b3d;

  /* Notebook specific styles */

  --jp-notebook-padding: 10px;
  --jp-notebook-select-background: var(--jp-layout-color1);
  --jp-notebook-multiselected-color: var(--md-blue-50);

  /* The scroll padding is calculated to fill enough space at the bottom of the
  notebook to show one single-line cell (with appropriate padding) at the top
  when the notebook is scrolled all the way to the bottom. We also subtract one
  pixel so that no scrollbar appears if we have just one single-line cell in the
  notebook. This padding is to enable a 'scroll past end' feature in a notebook.
  */
  --jp-notebook-scroll-padding: calc(
    100% - var(--jp-code-font-size) * var(--jp-code-line-height) -
      var(--jp-code-padding) - var(--jp-cell-padding) - 1px
  );

  /* Rendermime styles */

  --jp-rendermime-error-background: #fdd;
  --jp-rendermime-table-row-background: var(--md-grey-100);
  --jp-rendermime-table-row-hover-background: var(--md-light-blue-50);

  /* Dialog specific styles */

  --jp-dialog-background: rgba(0, 0, 0, 0.25);

  /* Console specific styles */

  --jp-console-padding: 10px;

  /* Toolbar specific styles */

  --jp-toolbar-border-color: var(--jp-border-color1);
  --jp-toolbar-micro-height: 8px;
  --jp-toolbar-background: var(--jp-layout-color1);
  --jp-toolbar-box-shadow: 0 0 2px 0 rgba(0, 0, 0, 0.24);
  --jp-toolbar-header-margin: 4px 4px 0 4px;
  --jp-toolbar-active-background: var(--md-grey-300);

  /* Statusbar specific styles */

  --jp-statusbar-height: 24px;

  /* Input field styles */

  --jp-input-box-shadow: inset 0 0 2px var(--md-blue-300);
  --jp-input-active-background: var(--jp-layout-color1);
  --jp-input-hover-background: var(--jp-layout-color1);
  --jp-input-background: var(--md-grey-100);
  --jp-input-border-color: var(--jp-inverse-border-color);
  --jp-input-active-border-color: var(--jp-brand-color1);
  --jp-input-active-box-shadow-color: rgba(19, 124, 189, 0.3);

  /* General editor styles */

  --jp-editor-selected-background: #d9d9d9;
  --jp-editor-selected-focused-background: #d7d4f0;
  --jp-editor-cursor-color: var(--jp-ui-font-color0);

  /* Code mirror specific styles */

  --jp-mirror-editor-keyword-color: #008000;
  --jp-mirror-editor-atom-color: #88f;
  --jp-mirror-editor-number-color: #080;
  --jp-mirror-editor-def-color: #00f;
  --jp-mirror-editor-variable-color: var(--md-grey-900);
  --jp-mirror-editor-variable-2-color: rgb(0, 54, 109);
  --jp-mirror-editor-variable-3-color: #085;
  --jp-mirror-editor-punctuation-color: #05a;
  --jp-mirror-editor-property-color: #05a;
  --jp-mirror-editor-operator-color: #a2f;
  --jp-mirror-editor-comment-color: #408080;
  --jp-mirror-editor-string-color: #ba2121;
  --jp-mirror-editor-string-2-color: #708;
  --jp-mirror-editor-meta-color: #a2f;
  --jp-mirror-editor-qualifier-color: #555;
  --jp-mirror-editor-builtin-color: #008000;
  --jp-mirror-editor-bracket-color: #997;
  --jp-mirror-editor-tag-color: #170;
  --jp-mirror-editor-attribute-color: #00c;
  --jp-mirror-editor-header-color: blue;
  --jp-mirror-editor-quote-color: #090;
  --jp-mirror-editor-link-color: #00c;
  --jp-mirror-editor-error-color: #f00;
  --jp-mirror-editor-hr-color: #999;

  /*
    RTC user specific colors.
    These colors are used for the cursor, username in the editor,
    and the icon of the user.
  */

  --jp-collaborator-color1: #ffad8e;
  --jp-collaborator-color2: #dac83d;
  --jp-collaborator-color3: #72dd76;
  --jp-collaborator-color4: #00e4d0;
  --jp-collaborator-color5: #45d4ff;
  --jp-collaborator-color6: #e2b1ff;
  --jp-collaborator-color7: #ff9de6;

  /* Vega extension styles */

  --jp-vega-background: white;

  /* Sidebar-related styles */

  --jp-sidebar-min-width: 250px;

  /* Search-related styles */

  --jp-search-toggle-off-opacity: 0.5;
  --jp-search-toggle-hover-opacity: 0.8;
  --jp-search-toggle-on-opacity: 1;
  --jp-search-selected-match-background-color: rgb(245, 200, 0);
  --jp-search-selected-match-color: black;
  --jp-search-unselected-match-background-color: var(
    --jp-inverse-layout-color0
  );
  --jp-search-unselected-match-color: var(--jp-ui-inverse-font-color0);

  /* Icon colors that work well with light or dark backgrounds */
  --jp-icon-contrast-color0: var(--md-purple-600);
  --jp-icon-contrast-color1: var(--md-green-600);
  --jp-icon-contrast-color2: var(--md-pink-600);
  --jp-icon-contrast-color3: var(--md-blue-600);

  /* Button colors */
  --jp-accept-color-normal: var(--md-blue-700);
  --jp-accept-color-hover: var(--md-blue-800);
  --jp-accept-color-active: var(--md-blue-900);
  --jp-warn-color-normal: var(--md-red-700);
  --jp-warn-color-hover: var(--md-red-800);
  --jp-warn-color-active: var(--md-red-900);
  --jp-reject-color-normal: var(--md-grey-600);
  --jp-reject-color-hover: var(--md-grey-700);
  --jp-reject-color-active: var(--md-grey-800);

  /* File or activity icons and switch semantic variables */
  --jp-jupyter-icon-color: #f37626;
  --jp-notebook-icon-color: #f37626;
  --jp-json-icon-color: var(--md-orange-700);
  --jp-console-icon-background-color: var(--md-blue-700);
  --jp-console-icon-color: white;
  --jp-terminal-icon-background-color: var(--md-grey-800);
  --jp-terminal-icon-color: var(--md-grey-200);
  --jp-text-editor-icon-color: var(--md-grey-700);
  --jp-inspector-icon-color: var(--md-grey-700);
  --jp-switch-color: var(--md-grey-400);
  --jp-switch-true-position-color: var(--md-orange-900);
}
</style>
<style type="text/css">
/* Force rendering true colors when outputing to pdf */
* {
  -webkit-print-color-adjust: exact;
}

/* Misc */
a.anchor-link {
  display: none;
}

/* Input area styling */
.jp-InputArea {
  overflow: hidden;
}

.jp-InputArea-editor {
  overflow: hidden;
}

.cm-editor.cm-s-jupyter .highlight pre {
/* weird, but --jp-code-padding defined to be 5px but 4px horizontal padding is hardcoded for pre.cm-line */
  padding: var(--jp-code-padding) 4px;
  margin: 0;

  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
  color: inherit;

}

.jp-OutputArea-output pre {
  line-height: inherit;
  font-family: inherit;
}

.jp-RenderedText pre {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-code-font-size);
}

/* Hiding the collapser by default */
.jp-Collapser {
  display: none;
}

@page {
    margin: 0.5in; /* Margin for each printed piece of paper */
}

@media print {
  .jp-Cell-inputWrapper,
  .jp-Cell-outputWrapper {
    display: block;
  }
}
</style>
<!-- Load mathjax -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS_CHTML-full,Safe"> </script>
<!-- MathJax configuration -->
<script type="text/x-mathjax-config">
    init_mathjax = function() {
        if (window.MathJax) {
        // MathJax loaded
            MathJax.Hub.Config({
                TeX: {
                    equationNumbers: {
                    autoNumber: "AMS",
                    useLabelIds: true
                    }
                },
                tex2jax: {
                    inlineMath: [ ['$','$'], ["\\(","\\)"] ],
                    displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
                    processEscapes: true,
                    processEnvironments: true
                },
                displayAlign: 'center',
                CommonHTML: {
                    linebreaks: {
                    automatic: true
                    }
                }
            });

            MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
        }
    }
    init_mathjax();
    </script>
<!-- End of mathjax configuration --><script type="module">
  document.addEventListener("DOMContentLoaded", async () => {
    const diagrams = document.querySelectorAll(".jp-Mermaid > pre.mermaid");
    // do not load mermaidjs if not needed
    if (!diagrams.length) {
      return;
    }
    const mermaid = (await import("https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.7.0/mermaid.esm.min.mjs")).default;
    const parser = new DOMParser();

    mermaid.initialize({
      maxTextSize: 100000,
      maxEdges: 100000,
      startOnLoad: false,
      fontFamily: window
        .getComputedStyle(document.body)
        .getPropertyValue("--jp-ui-font-family"),
      theme: document.querySelector("body[data-jp-theme-light='true']")
        ? "default"
        : "dark",
    });

    let _nextMermaidId = 0;

    function makeMermaidImage(svg) {
      const img = document.createElement("img");
      const doc = parser.parseFromString(svg, "image/svg+xml");
      const svgEl = doc.querySelector("svg");
      const { maxWidth } = svgEl?.style || {};
      const firstTitle = doc.querySelector("title");
      const firstDesc = doc.querySelector("desc");

      img.setAttribute("src", `data:image/svg+xml,${encodeURIComponent(svg)}`);
      if (maxWidth) {
        img.width = parseInt(maxWidth);
      }
      if (firstTitle) {
        img.setAttribute("alt", firstTitle.textContent);
      }
      if (firstDesc) {
        const caption = document.createElement("figcaption");
        caption.className = "sr-only";
        caption.textContent = firstDesc.textContent;
        return [img, caption];
      }
      return [img];
    }

    async function makeMermaidError(text) {
      let errorMessage = "";
      try {
        await mermaid.parse(text);
      } catch (err) {
        errorMessage = `${err}`;
      }

      const result = document.createElement("details");
      result.className = 'jp-RenderedMermaid-Details';
      const summary = document.createElement("summary");
      summary.className = 'jp-RenderedMermaid-Summary';
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.innerText = text;
      pre.appendChild(code);
      summary.appendChild(pre);
      result.appendChild(summary);

      const warning = document.createElement("pre");
      warning.innerText = errorMessage;
      result.appendChild(warning);
      return [result];
    }

    async function renderOneMarmaid(src) {
      const id = `jp-mermaid-${_nextMermaidId++}`;
      const parent = src.parentNode;
      let raw = src.textContent.trim();
      const el = document.createElement("div");
      el.style.visibility = "hidden";
      document.body.appendChild(el);
      let results = null;
      let output = null;
      try {
        let { svg } = await mermaid.render(id, raw, el);
        svg = cleanMermaidSvg(svg);
        results = makeMermaidImage(svg);
        output = document.createElement("figure");
        results.map(output.appendChild, output);
      } catch (err) {
        parent.classList.add("jp-mod-warning");
        results = await makeMermaidError(raw);
        output = results[0];
      } finally {
        el.remove();
      }
      parent.classList.add("jp-RenderedMermaid");
      parent.appendChild(output);
    }


    /**
     * Post-process to ensure mermaid diagrams contain only valid SVG and XHTML.
     */
    function cleanMermaidSvg(svg) {
      return svg.replace(RE_VOID_ELEMENT, replaceVoidElement);
    }


    /**
     * A regular expression for all void elements, which may include attributes and
     * a slash.
     *
     * @see https://developer.mozilla.org/en-US/docs/Glossary/Void_element
     *
     * Of these, only `<br>` is generated by Mermaid in place of `\n`,
     * but _any_ "malformed" tag will break the SVG rendering entirely.
     */
    const RE_VOID_ELEMENT =
      /<\s*(area|base|br|col|embed|hr|img|input|link|meta|param|source|track|wbr)\s*([^>]*?)\s*>/gi;

    /**
     * Ensure a void element is closed with a slash, preserving any attributes.
     */
    function replaceVoidElement(match, tag, rest) {
      rest = rest.trim();
      if (!rest.endsWith('/')) {
        rest = `${rest} /`;
      }
      return `<${tag} ${rest}>`;
    }

    void Promise.all([...diagrams].map(renderOneMarmaid));
  });
</script>
<style>
  .jp-Mermaid:not(.jp-RenderedMermaid) {
    display: none;
  }

  .jp-RenderedMermaid {
    overflow: auto;
    display: flex;
  }

  .jp-RenderedMermaid.jp-mod-warning {
    width: auto;
    padding: 0.5em;
    margin-top: 0.5em;
    border: var(--jp-border-width) solid var(--jp-warn-color2);
    border-radius: var(--jp-border-radius);
    color: var(--jp-ui-font-color1);
    font-size: var(--jp-ui-font-size1);
    white-space: pre-wrap;
    word-wrap: break-word;
  }

  .jp-RenderedMermaid figure {
    margin: 0;
    overflow: auto;
    max-width: 100%;
  }

  .jp-RenderedMermaid img {
    max-width: 100%;
  }

  .jp-RenderedMermaid-Details > pre {
    margin-top: 1em;
  }

  .jp-RenderedMermaid-Summary {
    color: var(--jp-warn-color2);
  }

  .jp-RenderedMermaid:not(.jp-mod-warning) pre {
    display: none;
  }

  .jp-RenderedMermaid-Summary > pre {
    display: inline-block;
    white-space: normal;
  }
</style>
<!-- End of mermaid configuration --></head>
<body class="jp-Notebook" data-jp-theme-light="true" data-jp-theme-name="JupyterLab Light">
<main><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [1]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>

<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">train_test_split</span>
<span class="kn">from</span> <span class="nn">sklearn.preprocessing</span> <span class="kn">import</span> <span class="n">StandardScaler</span>
<span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="kn">import</span> <span class="n">accuracy_score</span><span class="p">,</span> <span class="n">precision_score</span><span class="p">,</span> <span class="n">recall_score</span><span class="p">,</span> <span class="n">f1_score</span><span class="p">,</span> <span class="n">roc_auc_score</span><span class="p">,</span> <span class="n">classification_report</span>

<span class="kn">from</span> <span class="nn">sklearn.svm</span> <span class="kn">import</span> <span class="n">SVC</span>
<span class="kn">from</span> <span class="nn">sklearn.neighbors</span> <span class="kn">import</span> <span class="n">KNeighborsClassifier</span>
<span class="kn">from</span> <span class="nn">sklearn.naive_bayes</span> <span class="kn">import</span> <span class="n">GaussianNB</span>
<span class="kn">from</span> <span class="nn">sklearn.ensemble</span> <span class="kn">import</span> <span class="n">AdaBoostClassifier</span>
<span class="kn">from</span> <span class="nn">xgboost</span> <span class="kn">import</span> <span class="n">XGBClassifier</span>

<span class="kn">import</span> <span class="nn">warnings</span>
<span class="n">warnings</span><span class="o">.</span><span class="n">filterwarnings</span><span class="p">(</span><span class="s2">"ignore"</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [2]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Load your dataset</span>
<span class="n">df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_excel</span><span class="p">(</span><span class="s2">"/content/Earnings Manipulator (1).xlsx"</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [3]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child jp-OutputArea-executeResult">
<div class="jp-OutputPrompt jp-OutputArea-prompt">Out[3]:</div>
<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html" tabindex="0">
<div class="colab-df-container" id="df-3b1eaf29-10d2-479d-8c8a-e305e7077181">
<div>
<style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>Company ID</th>
<th>DSRI</th>
<th>GMI</th>
<th>AQI</th>
<th>SGI</th>
<th>DEPI</th>
<th>SGAI</th>
<th>ACCR</th>
<th>LEVI</th>
<th>Manipulator</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>1</td>
<td>1.624742</td>
<td>1.128927</td>
<td>7.185053</td>
<td>0.366211</td>
<td>1.381519</td>
<td>1.624145</td>
<td>-0.166809</td>
<td>1.161082</td>
<td>Yes</td>
</tr>
<tr>
<th>1</th>
<td>2</td>
<td>1.000000</td>
<td>1.606492</td>
<td>1.004988</td>
<td>13.081433</td>
<td>0.400000</td>
<td>5.198207</td>
<td>0.060475</td>
<td>0.986732</td>
<td>Yes</td>
</tr>
<tr>
<th>2</th>
<td>3</td>
<td>1.000000</td>
<td>1.015607</td>
<td>1.241389</td>
<td>1.475018</td>
<td>1.169353</td>
<td>0.647671</td>
<td>0.036732</td>
<td>1.264305</td>
<td>Yes</td>
</tr>
<tr>
<th>3</th>
<td>4</td>
<td>1.486239</td>
<td>1.000000</td>
<td>0.465535</td>
<td>0.672840</td>
<td>2.000000</td>
<td>0.092890</td>
<td>0.273434</td>
<td>0.680975</td>
<td>Yes</td>
</tr>
<tr>
<th>4</th>
<td>5</td>
<td>1.000000</td>
<td>1.369038</td>
<td>0.637112</td>
<td>0.861346</td>
<td>1.454676</td>
<td>1.741460</td>
<td>0.123048</td>
<td>0.939047</td>
<td>Yes</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-3b1eaf29-10d2-479d-8c8a-e305e7077181')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-3b1eaf29-10d2-479d-8c8a-e305e7077181 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-3b1eaf29-10d2-479d-8c8a-e305e7077181');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
<div id="df-5434806b-cfae-4343-b29e-fcbbb0556b37">
<button class="colab-df-quickchart" onclick="quickchart('df-5434806b-cfae-4343-b29e-fcbbb0556b37')" style="display:none;" title="Suggest charts">
<svg height="24px" viewbox="0 0 24 24" width="24px" xmlns="http://www.w3.org/2000/svg">
<g>
<path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"></path>
</g>
</svg>
</button>
<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>
<script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-5434806b-cfae-4343-b29e-fcbbb0556b37 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [4]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="n">df</span><span class="p">[[</span><span class="s1">'DSRI'</span><span class="p">,</span><span class="s1">'GMI'</span><span class="p">,</span><span class="s1">'AQI'</span><span class="p">,</span><span class="s1">'SGI'</span><span class="p">,</span><span class="s1">'DEPI'</span><span class="p">,</span><span class="s1">'SGAI'</span><span class="p">,</span><span class="s1">'ACCR'</span><span class="p">,</span><span class="s1">'LEVI'</span><span class="p">]]</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s1">'Manipulator'</span><span class="p">]</span><span class="o">.</span><span class="n">map</span><span class="p">({</span><span class="s1">'No'</span><span class="p">:</span> <span class="mi">0</span><span class="p">,</span> <span class="s1">'Yes'</span><span class="p">:</span> <span class="mi">1</span><span class="p">})</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [5]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span>
    <span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.25</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">42</span><span class="p">,</span> <span class="n">stratify</span><span class="o">=</span><span class="n">y</span>
<span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [6]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">scaler</span> <span class="o">=</span> <span class="n">StandardScaler</span><span class="p">()</span>
<span class="n">X_train_scaled</span> <span class="o">=</span> <span class="n">scaler</span><span class="o">.</span><span class="n">fit_transform</span><span class="p">(</span><span class="n">X_train</span><span class="p">)</span>
<span class="n">X_test_scaled</span> <span class="o">=</span> <span class="n">scaler</span><span class="o">.</span><span class="n">transform</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [7]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">svm_model</span> <span class="o">=</span> <span class="n">SVC</span><span class="p">(</span><span class="n">kernel</span><span class="o">=</span><span class="s1">'rbf'</span><span class="p">,</span> <span class="n">probability</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">svm_model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train_scaled</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
<span class="n">svm_pred</span> <span class="o">=</span> <span class="n">svm_model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test_scaled</span><span class="p">)</span>
<span class="n">svm_prob</span> <span class="o">=</span> <span class="n">svm_model</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">X_test_scaled</span><span class="p">)[:,</span><span class="mi">1</span><span class="p">]</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [8]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">knn_model</span> <span class="o">=</span> <span class="n">KNeighborsClassifier</span><span class="p">(</span><span class="n">n_neighbors</span><span class="o">=</span><span class="mi">5</span><span class="p">)</span>
<span class="n">knn_model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train_scaled</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
<span class="n">knn_pred</span> <span class="o">=</span> <span class="n">knn_model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test_scaled</span><span class="p">)</span>
<span class="n">knn_prob</span> <span class="o">=</span> <span class="n">knn_model</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">X_test_scaled</span><span class="p">)[:,</span><span class="mi">1</span><span class="p">]</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [9]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">nb_model</span> <span class="o">=</span> <span class="n">GaussianNB</span><span class="p">()</span>
<span class="n">nb_model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
<span class="n">nb_pred</span> <span class="o">=</span> <span class="n">nb_model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
<span class="n">nb_prob</span> <span class="o">=</span> <span class="n">nb_model</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">X_test</span><span class="p">)[:,</span><span class="mi">1</span><span class="p">]</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [10]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">ada_model</span> <span class="o">=</span> <span class="n">AdaBoostClassifier</span><span class="p">(</span><span class="n">n_estimators</span><span class="o">=</span><span class="mi">200</span><span class="p">,</span> <span class="n">learning_rate</span><span class="o">=</span><span class="mf">0.05</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">42</span><span class="p">)</span>
<span class="n">ada_model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
<span class="n">ada_pred</span> <span class="o">=</span> <span class="n">ada_model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
<span class="n">ada_prob</span> <span class="o">=</span> <span class="n">ada_model</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">X_test</span><span class="p">)[:,</span><span class="mi">1</span><span class="p">]</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [11]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">xgb_model</span> <span class="o">=</span> <span class="n">XGBClassifier</span><span class="p">(</span>
    <span class="n">n_estimators</span><span class="o">=</span><span class="mi">300</span><span class="p">,</span>
    <span class="n">learning_rate</span><span class="o">=</span><span class="mf">0.05</span><span class="p">,</span>
    <span class="n">max_depth</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">subsample</span><span class="o">=</span><span class="mf">0.9</span><span class="p">,</span>
    <span class="n">colsample_bytree</span><span class="o">=</span><span class="mf">0.9</span><span class="p">,</span>
    <span class="n">eval_metric</span><span class="o">=</span><span class="s1">'logloss'</span><span class="p">,</span>
    <span class="n">random_state</span><span class="o">=</span><span class="mi">42</span>
<span class="p">)</span>

<span class="n">xgb_model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
<span class="n">xgb_pred</span> <span class="o">=</span> <span class="n">xgb_model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
<span class="n">xgb_prob</span> <span class="o">=</span> <span class="n">xgb_model</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">X_test</span><span class="p">)[:,</span><span class="mi">1</span><span class="p">]</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [12]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">evaluate</span><span class="p">(</span><span class="n">model_name</span><span class="p">,</span> <span class="n">y_true</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">,</span> <span class="n">y_prob</span><span class="p">):</span>
    <span class="k">return</span> <span class="p">{</span>
        <span class="s2">"Model"</span><span class="p">:</span> <span class="n">model_name</span><span class="p">,</span>
        <span class="s2">"Accuracy"</span><span class="p">:</span> <span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_true</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">),</span>
        <span class="s2">"Precision"</span><span class="p">:</span> <span class="n">precision_score</span><span class="p">(</span><span class="n">y_true</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">),</span>
        <span class="s2">"Recall"</span><span class="p">:</span> <span class="n">recall_score</span><span class="p">(</span><span class="n">y_true</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">),</span>
        <span class="s2">"F1-score"</span><span class="p">:</span> <span class="n">f1_score</span><span class="p">(</span><span class="n">y_true</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">),</span>
        <span class="s2">"ROC-AUC"</span><span class="p">:</span> <span class="n">roc_auc_score</span><span class="p">(</span><span class="n">y_true</span><span class="p">,</span> <span class="n">y_prob</span><span class="p">)</span>
    <span class="p">}</span>

<span class="n">results</span> <span class="o">=</span> <span class="p">[]</span>

<span class="n">results</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">evaluate</span><span class="p">(</span><span class="s2">"SVM"</span><span class="p">,</span> <span class="n">y_test</span><span class="p">,</span> <span class="n">svm_pred</span><span class="p">,</span> <span class="n">svm_prob</span><span class="p">))</span>
<span class="n">results</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">evaluate</span><span class="p">(</span><span class="s2">"KNN"</span><span class="p">,</span> <span class="n">y_test</span><span class="p">,</span> <span class="n">knn_pred</span><span class="p">,</span> <span class="n">knn_prob</span><span class="p">))</span>
<span class="n">results</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">evaluate</span><span class="p">(</span><span class="s2">"Naive Bayes"</span><span class="p">,</span> <span class="n">y_test</span><span class="p">,</span> <span class="n">nb_pred</span><span class="p">,</span> <span class="n">nb_prob</span><span class="p">))</span>
<span class="n">results</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">evaluate</span><span class="p">(</span><span class="s2">"AdaBoost"</span><span class="p">,</span> <span class="n">y_test</span><span class="p">,</span> <span class="n">ada_pred</span><span class="p">,</span> <span class="n">ada_prob</span><span class="p">))</span>
<span class="n">results</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">evaluate</span><span class="p">(</span><span class="s2">"XGBoost"</span><span class="p">,</span> <span class="n">y_test</span><span class="p">,</span> <span class="n">xgb_pred</span><span class="p">,</span> <span class="n">xgb_prob</span><span class="p">))</span>

<span class="n">results_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">results</span><span class="p">)</span>
<span class="n">results_df</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child jp-OutputArea-executeResult">
<div class="jp-OutputPrompt jp-OutputArea-prompt">Out[12]:</div>
<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html" tabindex="0">
<div class="colab-df-container" id="df-5c2046d1-fd67-488e-bc2b-702f53d3dd89">
<div>
<style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>Model</th>
<th>Accuracy</th>
<th>Precision</th>
<th>Recall</th>
<th>F1-score</th>
<th>ROC-AUC</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>SVM</td>
<td>0.854545</td>
<td>1.000000</td>
<td>0.2</td>
<td>0.333333</td>
<td>0.862222</td>
</tr>
<tr>
<th>1</th>
<td>KNN</td>
<td>0.836364</td>
<td>1.000000</td>
<td>0.1</td>
<td>0.181818</td>
<td>0.781111</td>
</tr>
<tr>
<th>2</th>
<td>Naive Bayes</td>
<td>0.818182</td>
<td>0.500000</td>
<td>0.3</td>
<td>0.375000</td>
<td>0.724444</td>
</tr>
<tr>
<th>3</th>
<td>AdaBoost</td>
<td>0.836364</td>
<td>1.000000</td>
<td>0.1</td>
<td>0.181818</td>
<td>0.805556</td>
</tr>
<tr>
<th>4</th>
<td>XGBoost</td>
<td>0.890909</td>
<td>0.833333</td>
<td>0.5</td>
<td>0.625000</td>
<td>0.831111</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-5c2046d1-fd67-488e-bc2b-702f53d3dd89')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-5c2046d1-fd67-488e-bc2b-702f53d3dd89 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5c2046d1-fd67-488e-bc2b-702f53d3dd89');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
<div id="df-ad29cae6-7d06-45c2-83a1-10e4d9c61d41">
<button class="colab-df-quickchart" onclick="quickchart('df-ad29cae6-7d06-45c2-83a1-10e4d9c61d41')" style="display:none;" title="Suggest charts">
<svg height="24px" viewbox="0 0 24 24" width="24px" xmlns="http://www.w3.org/2000/svg">
<g>
<path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"></path>
</g>
</svg>
</button>
<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>
<script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-ad29cae6-7d06-45c2-83a1-10e4d9c61d41 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
</div>
<div id="id_94791d50-072b-4a0d-8d34-b840d95f2db0">
<style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
<button class="colab-df-generate" onclick="generateWithVariable('results_df')" style="display:none;" title="Generate code using this dataframe.">
<svg height="24px" viewbox="0 0 24 24" width="24px" xmlns="http://www.w3.org/2000/svg">
<path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"></path>
</svg>
</button>
<script>
      (() => {
      const buttonEl =
        document.querySelector('#id_94791d50-072b-4a0d-8d34-b840d95f2db0 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('results_df');
      }
      })();
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [13]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">test_sizes</span> <span class="o">=</span> <span class="p">[</span><span class="mf">0.30</span><span class="p">,</span> <span class="mf">0.25</span><span class="p">,</span> <span class="mf">0.20</span><span class="p">]</span>
<span class="n">all_splits_data</span> <span class="o">=</span> <span class="p">[]</span>

<span class="k">for</span> <span class="n">test_size</span> <span class="ow">in</span> <span class="n">test_sizes</span><span class="p">:</span>
    <span class="n">X_train_temp</span><span class="p">,</span> <span class="n">X_test_temp</span><span class="p">,</span> <span class="n">y_train_temp</span><span class="p">,</span> <span class="n">y_test_temp</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span>
        <span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="n">test_size</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">42</span><span class="p">,</span> <span class="n">stratify</span><span class="o">=</span><span class="n">y</span>
    <span class="p">)</span>

    <span class="n">scaler</span> <span class="o">=</span> <span class="n">StandardScaler</span><span class="p">()</span>
    <span class="n">X_train_scaled_temp</span> <span class="o">=</span> <span class="n">scaler</span><span class="o">.</span><span class="n">fit_transform</span><span class="p">(</span><span class="n">X_train_temp</span><span class="p">)</span>
    <span class="n">X_test_scaled_temp</span> <span class="o">=</span> <span class="n">scaler</span><span class="o">.</span><span class="n">transform</span><span class="p">(</span><span class="n">X_test_temp</span><span class="p">)</span>

    <span class="n">all_splits_data</span><span class="o">.</span><span class="n">append</span><span class="p">({</span>
        <span class="s2">"test_size"</span><span class="p">:</span> <span class="n">test_size</span><span class="p">,</span>
        <span class="s2">"X_train"</span><span class="p">:</span> <span class="n">X_train_temp</span><span class="p">,</span>
        <span class="s2">"X_test"</span><span class="p">:</span> <span class="n">X_test_temp</span><span class="p">,</span>
        <span class="s2">"y_train"</span><span class="p">:</span> <span class="n">y_train_temp</span><span class="p">,</span>
        <span class="s2">"y_test"</span><span class="p">:</span> <span class="n">y_test_temp</span><span class="p">,</span>
        <span class="s2">"X_train_scaled"</span><span class="p">:</span> <span class="n">X_train_scaled_temp</span><span class="p">,</span>
        <span class="s2">"X_test_scaled"</span><span class="p">:</span> <span class="n">X_test_scaled_temp</span>
    <span class="p">})</span>

<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Generated </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">all_splits_data</span><span class="p">)</span><span class="si">}</span><span class="s2"> train-test splits with scaled data."</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Example: First split test size is </span><span class="si">{</span><span class="n">all_splits_data</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="s1">'test_size'</span><span class="p">]</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Generated 3 train-test splits with scaled data.
Example: First split test size is 0.3
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [14]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">GridSearchCV</span>

<span class="c1"># Define parameter grid for SVM</span>
<span class="n">param_grid_svm</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s1">'C'</span><span class="p">:</span> <span class="p">[</span><span class="mf">0.1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">10</span><span class="p">,</span> <span class="mi">100</span><span class="p">],</span>
    <span class="s1">'kernel'</span><span class="p">:</span> <span class="p">[</span><span class="s1">'linear'</span><span class="p">,</span> <span class="s1">'rbf'</span><span class="p">],</span>
    <span class="s1">'gamma'</span><span class="p">:</span> <span class="p">[</span><span class="s1">'scale'</span><span class="p">,</span> <span class="s1">'auto'</span><span class="p">,</span> <span class="mf">0.1</span><span class="p">,</span> <span class="mi">1</span><span class="p">]</span>
<span class="p">}</span>

<span class="n">svm_tuning_results</span> <span class="o">=</span> <span class="p">[]</span>

<span class="k">for</span> <span class="n">split_data</span> <span class="ow">in</span> <span class="n">all_splits_data</span><span class="p">:</span>
    <span class="n">test_size</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'test_size'</span><span class="p">]</span>
    <span class="n">X_train_scaled</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'X_train_scaled'</span><span class="p">]</span>
    <span class="n">X_test_scaled</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'X_test_scaled'</span><span class="p">]</span>
    <span class="n">y_train</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'y_train'</span><span class="p">]</span>
    <span class="n">y_test</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'y_test'</span><span class="p">]</span>

    <span class="c1"># Instantiate SVM model</span>
    <span class="n">svm_model</span> <span class="o">=</span> <span class="n">SVC</span><span class="p">(</span><span class="n">probability</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">42</span><span class="p">)</span>

    <span class="c1"># Create GridSearchCV object</span>
    <span class="n">grid_search_svm</span> <span class="o">=</span> <span class="n">GridSearchCV</span><span class="p">(</span><span class="n">svm_model</span><span class="p">,</span> <span class="n">param_grid_svm</span><span class="p">,</span> <span class="n">cv</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">scoring</span><span class="o">=</span><span class="s1">'roc_auc'</span><span class="p">,</span> <span class="n">refit</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">n_jobs</span><span class="o">=-</span><span class="mi">1</span><span class="p">,</span> <span class="n">verbose</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>

    <span class="c1"># Fit GridSearchCV</span>
    <span class="n">grid_search_svm</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train_scaled</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>

    <span class="c1"># Get the best estimator</span>
    <span class="n">best_svm_estimator</span> <span class="o">=</span> <span class="n">grid_search_svm</span><span class="o">.</span><span class="n">best_estimator_</span>

    <span class="c1"># Make predictions and obtain probabilities</span>
    <span class="n">svm_pred_tuned</span> <span class="o">=</span> <span class="n">best_svm_estimator</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test_scaled</span><span class="p">)</span>
    <span class="n">svm_prob_tuned</span> <span class="o">=</span> <span class="n">best_svm_estimator</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">X_test_scaled</span><span class="p">)[:,</span> <span class="mi">1</span><span class="p">]</span>

    <span class="c1"># Evaluate the best estimator</span>
    <span class="n">evaluation_metrics</span> <span class="o">=</span> <span class="n">evaluate</span><span class="p">(</span><span class="s2">"SVM_tuned"</span><span class="p">,</span> <span class="n">y_test</span><span class="p">,</span> <span class="n">svm_pred_tuned</span><span class="p">,</span> <span class="n">svm_prob_tuned</span><span class="p">)</span>

    <span class="c1"># Add best parameters and test size to the results</span>
    <span class="n">evaluation_metrics</span><span class="p">[</span><span class="s1">'Best Parameters'</span><span class="p">]</span> <span class="o">=</span> <span class="n">grid_search_svm</span><span class="o">.</span><span class="n">best_params_</span>
    <span class="n">evaluation_metrics</span><span class="p">[</span><span class="s1">'Test Size'</span><span class="p">]</span> <span class="o">=</span> <span class="n">test_size</span>
    <span class="n">svm_tuning_results</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">evaluation_metrics</span><span class="p">)</span>

<span class="c1"># Display the results</span>
<span class="n">svm_tuning_results_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">svm_tuning_results</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"SVM Hyperparameter Tuning Results:"</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">svm_tuning_results_df</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>SVM Hyperparameter Tuning Results:
       Model  Accuracy  Precision  Recall  F1-score   ROC-AUC  \
0  SVM_tuned  0.818182   0.000000     0.0  0.000000  0.811728   
1  SVM_tuned  0.836364   0.666667     0.2  0.307692  0.893333   
2  SVM_tuned  0.818182   0.000000     0.0  0.000000  0.843750   

                                 Best Parameters  Test Size  
0        {'C': 0.1, 'gamma': 1, 'kernel': 'rbf'}       0.30  
1       {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}       0.25  
2  {'C': 0.1, 'gamma': 'scale', 'kernel': 'rbf'}       0.20  
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [15]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">GridSearchCV</span>

<span class="c1"># Define parameter grid for KNN</span>
<span class="n">param_grid_knn</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s1">'n_neighbors'</span><span class="p">:</span> <span class="p">[</span><span class="mi">3</span><span class="p">,</span> <span class="mi">5</span><span class="p">,</span> <span class="mi">7</span><span class="p">,</span> <span class="mi">9</span><span class="p">,</span> <span class="mi">11</span><span class="p">],</span>
    <span class="s1">'weights'</span><span class="p">:</span> <span class="p">[</span><span class="s1">'uniform'</span><span class="p">,</span> <span class="s1">'distance'</span><span class="p">],</span>
    <span class="s1">'metric'</span><span class="p">:</span> <span class="p">[</span><span class="s1">'euclidean'</span><span class="p">,</span> <span class="s1">'manhattan'</span><span class="p">,</span> <span class="s1">'minkowski'</span><span class="p">]</span>
<span class="p">}</span>

<span class="n">knn_tuning_results</span> <span class="o">=</span> <span class="p">[]</span>

<span class="k">for</span> <span class="n">split_data</span> <span class="ow">in</span> <span class="n">all_splits_data</span><span class="p">:</span>
    <span class="n">test_size</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'test_size'</span><span class="p">]</span>
    <span class="n">X_train_scaled</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'X_train_scaled'</span><span class="p">]</span>
    <span class="n">X_test_scaled</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'X_test_scaled'</span><span class="p">]</span>
    <span class="n">y_train</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'y_train'</span><span class="p">]</span>
    <span class="n">y_test</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'y_test'</span><span class="p">]</span>

    <span class="c1"># Instantiate KNN model</span>
    <span class="n">knn_model</span> <span class="o">=</span> <span class="n">KNeighborsClassifier</span><span class="p">()</span>

    <span class="c1"># Create GridSearchCV object</span>
    <span class="n">grid_search_knn</span> <span class="o">=</span> <span class="n">GridSearchCV</span><span class="p">(</span><span class="n">knn_model</span><span class="p">,</span> <span class="n">param_grid_knn</span><span class="p">,</span> <span class="n">cv</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">scoring</span><span class="o">=</span><span class="s1">'roc_auc'</span><span class="p">,</span> <span class="n">refit</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">n_jobs</span><span class="o">=-</span><span class="mi">1</span><span class="p">,</span> <span class="n">verbose</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>

    <span class="c1"># Fit GridSearchCV</span>
    <span class="n">grid_search_knn</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train_scaled</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>

    <span class="c1"># Get the best estimator</span>
    <span class="n">best_knn_estimator</span> <span class="o">=</span> <span class="n">grid_search_knn</span><span class="o">.</span><span class="n">best_estimator_</span>

    <span class="c1"># Make predictions and obtain probabilities</span>
    <span class="n">knn_pred_tuned</span> <span class="o">=</span> <span class="n">best_knn_estimator</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test_scaled</span><span class="p">)</span>
    <span class="n">knn_prob_tuned</span> <span class="o">=</span> <span class="n">best_knn_estimator</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">X_test_scaled</span><span class="p">)[:,</span> <span class="mi">1</span><span class="p">]</span>

    <span class="c1"># Evaluate the best estimator</span>
    <span class="n">evaluation_metrics</span> <span class="o">=</span> <span class="n">evaluate</span><span class="p">(</span><span class="s2">"KNN_tuned"</span><span class="p">,</span> <span class="n">y_test</span><span class="p">,</span> <span class="n">knn_pred_tuned</span><span class="p">,</span> <span class="n">knn_prob_tuned</span><span class="p">)</span>

    <span class="c1"># Add best parameters and test size to the results</span>
    <span class="n">evaluation_metrics</span><span class="p">[</span><span class="s1">'Best Parameters'</span><span class="p">]</span> <span class="o">=</span> <span class="n">grid_search_knn</span><span class="o">.</span><span class="n">best_params_</span>
    <span class="n">evaluation_metrics</span><span class="p">[</span><span class="s1">'Test Size'</span><span class="p">]</span> <span class="o">=</span> <span class="n">test_size</span>
    <span class="n">knn_tuning_results</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">evaluation_metrics</span><span class="p">)</span>

<span class="c1"># Display the results</span>
<span class="n">knn_tuning_results_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">knn_tuning_results</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"KNN Hyperparameter Tuning Results:"</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">knn_tuning_results_df</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>KNN Hyperparameter Tuning Results:
       Model  Accuracy  Precision  Recall  F1-score   ROC-AUC  \
0  KNN_tuned  0.818182        0.0   0.000  0.000000  0.754630   
1  KNN_tuned  0.836364        1.0   0.100  0.181818  0.820000   
2  KNN_tuned  0.840909        1.0   0.125  0.222222  0.866319   

                                     Best Parameters  Test Size  
0  {'metric': 'manhattan', 'n_neighbors': 11, 'we...       0.30  
1  {'metric': 'euclidean', 'n_neighbors': 11, 'we...       0.25  
2  {'metric': 'euclidean', 'n_neighbors': 11, 'we...       0.20  
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [16]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">GridSearchCV</span>

<span class="c1"># Define parameter grid for Naive Bayes</span>
<span class="n">param_grid_nb</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s1">'var_smoothing'</span><span class="p">:</span> <span class="n">np</span><span class="o">.</span><span class="n">logspace</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="o">-</span><span class="mi">9</span><span class="p">,</span> <span class="n">num</span><span class="o">=</span><span class="mi">10</span><span class="p">)</span>
<span class="p">}</span>

<span class="n">nb_tuning_results</span> <span class="o">=</span> <span class="p">[]</span>

<span class="k">for</span> <span class="n">split_data</span> <span class="ow">in</span> <span class="n">all_splits_data</span><span class="p">:</span>
    <span class="n">test_size</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'test_size'</span><span class="p">]</span>
    <span class="c1"># GaussianNB does not typically require scaled data, so use unscaled X_train and X_test</span>
    <span class="n">X_train</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'X_train'</span><span class="p">]</span>
    <span class="n">X_test</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'X_test'</span><span class="p">]</span>
    <span class="n">y_train</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'y_train'</span><span class="p">]</span>
    <span class="n">y_test</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'y_test'</span><span class="p">]</span>

    <span class="c1"># Instantiate GaussianNB model</span>
    <span class="n">nb_model</span> <span class="o">=</span> <span class="n">GaussianNB</span><span class="p">()</span>

    <span class="c1"># Create GridSearchCV object</span>
    <span class="n">grid_search_nb</span> <span class="o">=</span> <span class="n">GridSearchCV</span><span class="p">(</span><span class="n">nb_model</span><span class="p">,</span> <span class="n">param_grid_nb</span><span class="p">,</span> <span class="n">cv</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">scoring</span><span class="o">=</span><span class="s1">'roc_auc'</span><span class="p">,</span> <span class="n">refit</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">n_jobs</span><span class="o">=-</span><span class="mi">1</span><span class="p">,</span> <span class="n">verbose</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>

    <span class="c1"># Fit GridSearchCV</span>
    <span class="n">grid_search_nb</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>

    <span class="c1"># Get the best estimator</span>
    <span class="n">best_nb_estimator</span> <span class="o">=</span> <span class="n">grid_search_nb</span><span class="o">.</span><span class="n">best_estimator_</span>

    <span class="c1"># Make predictions and obtain probabilities</span>
    <span class="n">nb_pred_tuned</span> <span class="o">=</span> <span class="n">best_nb_estimator</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
    <span class="n">nb_prob_tuned</span> <span class="o">=</span> <span class="n">best_nb_estimator</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">X_test</span><span class="p">)[:,</span> <span class="mi">1</span><span class="p">]</span>

    <span class="c1"># Evaluate the best estimator</span>
    <span class="n">evaluation_metrics</span> <span class="o">=</span> <span class="n">evaluate</span><span class="p">(</span><span class="s2">"Naive Bayes_tuned"</span><span class="p">,</span> <span class="n">y_test</span><span class="p">,</span> <span class="n">nb_pred_tuned</span><span class="p">,</span> <span class="n">nb_prob_tuned</span><span class="p">)</span>

    <span class="c1"># Add best parameters and test size to the results</span>
    <span class="n">evaluation_metrics</span><span class="p">[</span><span class="s1">'Best Parameters'</span><span class="p">]</span> <span class="o">=</span> <span class="n">grid_search_nb</span><span class="o">.</span><span class="n">best_params_</span>
    <span class="n">evaluation_metrics</span><span class="p">[</span><span class="s1">'Test Size'</span><span class="p">]</span> <span class="o">=</span> <span class="n">test_size</span>
    <span class="n">nb_tuning_results</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">evaluation_metrics</span><span class="p">)</span>

<span class="c1"># Display the results</span>
<span class="n">nb_tuning_results_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">nb_tuning_results</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"Naive Bayes Hyperparameter Tuning Results:"</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">nb_tuning_results_df</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Naive Bayes Hyperparameter Tuning Results:
               Model  Accuracy  Precision    Recall  F1-score   ROC-AUC  \
0  Naive Bayes_tuned  0.833333   0.571429  0.333333  0.421053  0.743827   
1  Naive Bayes_tuned  0.818182   0.500000  0.300000  0.375000  0.724444   
2  Naive Bayes_tuned  0.840909   0.600000  0.375000  0.461538  0.743056   

            Best Parameters  Test Size  
0  {'var_smoothing': 1e-06}       0.30  
1  {'var_smoothing': 1e-06}       0.25  
2  {'var_smoothing': 1e-05}       0.20  
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [17]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">GridSearchCV</span>

<span class="c1"># Define parameter grid for AdaBoost</span>
<span class="n">param_grid_ada</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s1">'n_estimators'</span><span class="p">:</span> <span class="p">[</span><span class="mi">50</span><span class="p">,</span> <span class="mi">100</span><span class="p">,</span> <span class="mi">200</span><span class="p">],</span>
    <span class="s1">'learning_rate'</span><span class="p">:</span> <span class="p">[</span><span class="mf">0.01</span><span class="p">,</span> <span class="mf">0.05</span><span class="p">,</span> <span class="mf">0.1</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">]</span>
<span class="p">}</span>

<span class="n">ada_tuning_results</span> <span class="o">=</span> <span class="p">[]</span>

<span class="k">for</span> <span class="n">split_data</span> <span class="ow">in</span> <span class="n">all_splits_data</span><span class="p">:</span>
    <span class="n">test_size</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'test_size'</span><span class="p">]</span>
    <span class="c1"># AdaBoost typically does not require scaled data, so use unscaled X_train and X_test</span>
    <span class="n">X_train</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'X_train'</span><span class="p">]</span>
    <span class="n">X_test</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'X_test'</span><span class="p">]</span>
    <span class="n">y_train</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'y_train'</span><span class="p">]</span>
    <span class="n">y_test</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'y_test'</span><span class="p">]</span>

    <span class="c1"># Instantiate AdaBoost model</span>
    <span class="n">ada_model</span> <span class="o">=</span> <span class="n">AdaBoostClassifier</span><span class="p">(</span><span class="n">random_state</span><span class="o">=</span><span class="mi">42</span><span class="p">)</span>

    <span class="c1"># Create GridSearchCV object</span>
    <span class="n">grid_search_ada</span> <span class="o">=</span> <span class="n">GridSearchCV</span><span class="p">(</span><span class="n">ada_model</span><span class="p">,</span> <span class="n">param_grid_ada</span><span class="p">,</span> <span class="n">cv</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">scoring</span><span class="o">=</span><span class="s1">'roc_auc'</span><span class="p">,</span> <span class="n">refit</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">n_jobs</span><span class="o">=-</span><span class="mi">1</span><span class="p">,</span> <span class="n">verbose</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>

    <span class="c1"># Fit GridSearchCV</span>
    <span class="n">grid_search_ada</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>

    <span class="c1"># Get the best estimator</span>
    <span class="n">best_ada_estimator</span> <span class="o">=</span> <span class="n">grid_search_ada</span><span class="o">.</span><span class="n">best_estimator_</span>

    <span class="c1"># Make predictions and obtain probabilities</span>
    <span class="n">ada_pred_tuned</span> <span class="o">=</span> <span class="n">best_ada_estimator</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
    <span class="n">ada_prob_tuned</span> <span class="o">=</span> <span class="n">best_ada_estimator</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">X_test</span><span class="p">)[:,</span> <span class="mi">1</span><span class="p">]</span>

    <span class="c1"># Evaluate the best estimator</span>
    <span class="n">evaluation_metrics</span> <span class="o">=</span> <span class="n">evaluate</span><span class="p">(</span><span class="s2">"AdaBoost_tuned"</span><span class="p">,</span> <span class="n">y_test</span><span class="p">,</span> <span class="n">ada_pred_tuned</span><span class="p">,</span> <span class="n">ada_prob_tuned</span><span class="p">)</span>

    <span class="c1"># Add best parameters and test size to the results</span>
    <span class="n">evaluation_metrics</span><span class="p">[</span><span class="s1">'Best Parameters'</span><span class="p">]</span> <span class="o">=</span> <span class="n">grid_search_ada</span><span class="o">.</span><span class="n">best_params_</span>
    <span class="n">evaluation_metrics</span><span class="p">[</span><span class="s1">'Test Size'</span><span class="p">]</span> <span class="o">=</span> <span class="n">test_size</span>
    <span class="n">ada_tuning_results</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">evaluation_metrics</span><span class="p">)</span>

<span class="c1"># Display the results</span>
<span class="n">ada_tuning_results_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">ada_tuning_results</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"AdaBoost Hyperparameter Tuning Results:"</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">ada_tuning_results_df</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>AdaBoost Hyperparameter Tuning Results:
            Model  Accuracy  Precision  Recall  F1-score   ROC-AUC  \
0  AdaBoost_tuned  0.848485       0.75   0.250  0.375000  0.818673   
1  AdaBoost_tuned  0.854545       0.75   0.300  0.428571  0.857778   
2  AdaBoost_tuned  0.931818       1.00   0.625  0.769231  0.906250   

                               Best Parameters  Test Size  
0  {'learning_rate': 0.5, 'n_estimators': 200}       0.30  
1  {'learning_rate': 0.5, 'n_estimators': 200}       0.25  
2  {'learning_rate': 0.5, 'n_estimators': 200}       0.20  
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [18]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">GridSearchCV</span>

<span class="c1"># Define parameter grid for XGBoost</span>
<span class="n">param_grid_xgb</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s1">'n_estimators'</span><span class="p">:</span> <span class="p">[</span><span class="mi">100</span><span class="p">,</span> <span class="mi">200</span><span class="p">,</span> <span class="mi">300</span><span class="p">],</span>
    <span class="s1">'learning_rate'</span><span class="p">:</span> <span class="p">[</span><span class="mf">0.01</span><span class="p">,</span> <span class="mf">0.05</span><span class="p">,</span> <span class="mf">0.1</span><span class="p">],</span>
    <span class="s1">'max_depth'</span><span class="p">:</span> <span class="p">[</span><span class="mi">3</span><span class="p">,</span> <span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">],</span>
    <span class="s1">'subsample'</span><span class="p">:</span> <span class="p">[</span><span class="mf">0.8</span><span class="p">,</span> <span class="mf">0.9</span><span class="p">,</span> <span class="mf">1.0</span><span class="p">],</span>
    <span class="s1">'colsample_bytree'</span><span class="p">:</span> <span class="p">[</span><span class="mf">0.8</span><span class="p">,</span> <span class="mf">0.9</span><span class="p">,</span> <span class="mf">1.0</span><span class="p">]</span>
<span class="p">}</span>

<span class="n">xgb_tuning_results</span> <span class="o">=</span> <span class="p">[]</span>

<span class="k">for</span> <span class="n">split_data</span> <span class="ow">in</span> <span class="n">all_splits_data</span><span class="p">:</span>
    <span class="n">test_size</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'test_size'</span><span class="p">]</span>
    <span class="c1"># XGBoost does not typically require scaled data, so use unscaled X_train and X_test</span>
    <span class="n">X_train</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'X_train'</span><span class="p">]</span>
    <span class="n">X_test</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'X_test'</span><span class="p">]</span>
    <span class="n">y_train</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'y_train'</span><span class="p">]</span>
    <span class="n">y_test</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'y_test'</span><span class="p">]</span>

    <span class="c1"># Instantiate XGBoost model</span>
    <span class="n">xgb_model</span> <span class="o">=</span> <span class="n">XGBClassifier</span><span class="p">(</span><span class="n">random_state</span><span class="o">=</span><span class="mi">42</span><span class="p">,</span> <span class="n">eval_metric</span><span class="o">=</span><span class="s1">'logloss'</span><span class="p">,</span> <span class="n">use_label_encoder</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>

    <span class="c1"># Create GridSearchCV object</span>
    <span class="n">grid_search_xgb</span> <span class="o">=</span> <span class="n">GridSearchCV</span><span class="p">(</span><span class="n">xgb_model</span><span class="p">,</span> <span class="n">param_grid_xgb</span><span class="p">,</span> <span class="n">cv</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">scoring</span><span class="o">=</span><span class="s1">'roc_auc'</span><span class="p">,</span> <span class="n">refit</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">n_jobs</span><span class="o">=-</span><span class="mi">1</span><span class="p">,</span> <span class="n">verbose</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>

    <span class="c1"># Fit GridSearchCV</span>
    <span class="n">grid_search_xgb</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>

    <span class="c1"># Get the best estimator</span>
    <span class="n">best_xgb_estimator</span> <span class="o">=</span> <span class="n">grid_search_xgb</span><span class="o">.</span><span class="n">best_estimator_</span>

    <span class="c1"># Make predictions and obtain probabilities</span>
    <span class="n">xgb_pred_tuned</span> <span class="o">=</span> <span class="n">best_xgb_estimator</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
    <span class="n">xgb_prob_tuned</span> <span class="o">=</span> <span class="n">best_xgb_estimator</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">X_test</span><span class="p">)[:,</span> <span class="mi">1</span><span class="p">]</span>

    <span class="c1"># Evaluate the best estimator</span>
    <span class="n">evaluation_metrics</span> <span class="o">=</span> <span class="n">evaluate</span><span class="p">(</span><span class="s2">"XGBoost_tuned"</span><span class="p">,</span> <span class="n">y_test</span><span class="p">,</span> <span class="n">xgb_pred_tuned</span><span class="p">,</span> <span class="n">xgb_prob_tuned</span><span class="p">)</span>

    <span class="c1"># Add best parameters and test size to the results</span>
    <span class="n">evaluation_metrics</span><span class="p">[</span><span class="s1">'Best Parameters'</span><span class="p">]</span> <span class="o">=</span> <span class="n">grid_search_xgb</span><span class="o">.</span><span class="n">best_params_</span>
    <span class="n">evaluation_metrics</span><span class="p">[</span><span class="s1">'Test Size'</span><span class="p">]</span> <span class="o">=</span> <span class="n">test_size</span>
    <span class="n">xgb_tuning_results</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">evaluation_metrics</span><span class="p">)</span>

<span class="c1"># Display the results</span>
<span class="n">xgb_tuning_results_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">xgb_tuning_results</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"XGBoost Hyperparameter Tuning Results:"</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">xgb_tuning_results_df</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>XGBoost Hyperparameter Tuning Results:
           Model  Accuracy  Precision  Recall  F1-score   ROC-AUC  \
0  XGBoost_tuned  0.818182        0.0   0.000  0.000000  0.745370   
1  XGBoost_tuned  0.872727        0.8   0.400  0.533333  0.833333   
2  XGBoost_tuned  0.931818        1.0   0.625  0.769231  0.829861   

                                     Best Parameters  Test Size  
0  {'colsample_bytree': 0.8, 'learning_rate': 0.0...       0.30  
1  {'colsample_bytree': 0.8, 'learning_rate': 0.0...       0.25  
2  {'colsample_bytree': 0.8, 'learning_rate': 0.0...       0.20  
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [21]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>

<span class="n">best_svm</span> <span class="o">=</span> <span class="n">svm_tuning_results_df</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">svm_tuning_results_df</span><span class="p">[</span><span class="s1">'Accuracy'</span><span class="p">]</span><span class="o">.</span><span class="n">idxmax</span><span class="p">()]</span><span class="o">.</span><span class="n">to_dict</span><span class="p">()</span>
<span class="n">best_knn</span> <span class="o">=</span> <span class="n">knn_tuning_results_df</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">knn_tuning_results_df</span><span class="p">[</span><span class="s1">'Accuracy'</span><span class="p">]</span><span class="o">.</span><span class="n">idxmax</span><span class="p">()]</span><span class="o">.</span><span class="n">to_dict</span><span class="p">()</span>
<span class="n">best_nb</span> <span class="o">=</span> <span class="n">nb_tuning_results_df</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">nb_tuning_results_df</span><span class="p">[</span><span class="s1">'Accuracy'</span><span class="p">]</span><span class="o">.</span><span class="n">idxmax</span><span class="p">()]</span><span class="o">.</span><span class="n">to_dict</span><span class="p">()</span>
<span class="n">best_ada</span> <span class="o">=</span> <span class="n">ada_tuning_results_df</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">ada_tuning_results_df</span><span class="p">[</span><span class="s1">'Accuracy'</span><span class="p">]</span><span class="o">.</span><span class="n">idxmax</span><span class="p">()]</span><span class="o">.</span><span class="n">to_dict</span><span class="p">()</span>
<span class="n">best_xgb</span> <span class="o">=</span> <span class="n">xgb_tuning_results_df</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">xgb_tuning_results_df</span><span class="p">[</span><span class="s1">'Accuracy'</span><span class="p">]</span><span class="o">.</span><span class="n">idxmax</span><span class="p">()]</span><span class="o">.</span><span class="n">to_dict</span><span class="p">()</span>

<span class="n">beneish_rule</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s1">'Model'</span><span class="p">:</span> <span class="s1">'Beneish (rule)'</span><span class="p">,</span>
    <span class="s1">'Accuracy'</span><span class="p">:</span> <span class="mf">0.836364</span><span class="p">,</span>
    <span class="s1">'Precision'</span><span class="p">:</span> <span class="mf">0.555556</span><span class="p">,</span>
    <span class="s1">'Recall'</span><span class="p">:</span> <span class="mf">0.5</span><span class="p">,</span>
    <span class="s1">'F1-score'</span><span class="p">:</span> <span class="mf">0.526316</span><span class="p">,</span>
    <span class="s1">'ROC-AUC'</span><span class="p">:</span> <span class="mf">0.904444</span><span class="p">,</span>
    <span class="s1">'Best Parameters'</span><span class="p">:</span> <span class="s1">'N/A'</span><span class="p">,</span>
    <span class="s1">'Test Size'</span><span class="p">:</span> <span class="s1">'N/A'</span>
<span class="p">}</span>

<span class="n">best_results</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">([</span>
    <span class="n">best_svm</span><span class="p">,</span>
    <span class="n">best_knn</span><span class="p">,</span>
    <span class="n">best_nb</span><span class="p">,</span>
    <span class="n">best_ada</span><span class="p">,</span>
    <span class="n">best_xgb</span><span class="p">,</span>
    <span class="n">beneish_rule</span>
<span class="p">])</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">"</span><span class="se">\n</span><span class="s2">Overall Best Performing Models (Highest Accuracy per Model Type):</span><span class="se">\n</span><span class="s2">"</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">best_results</span><span class="p">[[</span><span class="s1">'Model'</span><span class="p">,</span> <span class="s1">'Accuracy'</span><span class="p">,</span> <span class="s1">'Precision'</span><span class="p">,</span> <span class="s1">'Recall'</span><span class="p">,</span> <span class="s1">'F1-score'</span><span class="p">,</span> <span class="s1">'ROC-AUC'</span><span class="p">]])</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>
Overall Best Performing Models (Highest Accuracy per Model Type):

               Model  Accuracy  Precision  Recall  F1-score   ROC-AUC
0          SVM_tuned  0.836364   0.666667   0.200  0.307692  0.893333
1          KNN_tuned  0.840909   1.000000   0.125  0.222222  0.866319
2  Naive Bayes_tuned  0.840909   0.600000   0.375  0.461538  0.743056
3     AdaBoost_tuned  0.931818   1.000000   0.625  0.769231  0.906250
4      XGBoost_tuned  0.931818   1.000000   0.625  0.769231  0.829861
5     Beneish (rule)  0.836364   0.555556   0.500  0.526316  0.904444
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h2 id="Insights-on-Best-Performing-Models">Insights on Best Performing Models<a class="anchor-link" href="#Insights-on-Best-Performing-Models">¶</a></h2><p>Based on the overall best performing models, both <strong>AdaBoost_tuned</strong> and <strong>XGBoost_tuned</strong> achieved the highest accuracy of <strong>0.931818</strong>. They also share the same Precision (1.00), Recall (0.625), and F1-score (0.769231).</p>
<p>However, <strong>AdaBoost_tuned</strong> has a slightly higher <strong>ROC-AUC of 0.906250</strong> compared to XGBoost_tuned's 0.829861. A higher ROC-AUC indicates better overall discriminative power across different classification thresholds, which is often a more robust measure for imbalanced datasets.</p>
<p>In comparison to the baseline <strong>Beneish (rule)</strong>, which had an Accuracy of 0.836364 and ROC-AUC of 0.904444, both AdaBoost and XGBoost tuned models show significant improvements in Accuracy and F1-score, while AdaBoost also slightly surpasses Beneish in ROC-AUC.</p>
<p>Therefore, <strong>AdaBoost_tuned</strong> can be considered the overall best performing model among the evaluated algorithms when considering a balance of accuracy, F1-score, and ROC-AUC.</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [23]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">shap</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>

<span class="c1"># Get the best XGBoost parameters and test size from the best_xgb dictionary</span>
<span class="n">best_xgb_params</span> <span class="o">=</span> <span class="n">best_xgb</span><span class="p">[</span><span class="s1">'Best Parameters'</span><span class="p">]</span>
<span class="n">best_xgb_test_size</span> <span class="o">=</span> <span class="n">best_xgb</span><span class="p">[</span><span class="s1">'Test Size'</span><span class="p">]</span>

<span class="c1"># Find the corresponding training data for the best test size</span>
<span class="n">X_train_for_shap</span> <span class="o">=</span> <span class="kc">None</span>
<span class="n">y_train_for_shap</span> <span class="o">=</span> <span class="kc">None</span>

<span class="k">for</span> <span class="n">split_data</span> <span class="ow">in</span> <span class="n">all_splits_data</span><span class="p">:</span>
    <span class="k">if</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'test_size'</span><span class="p">]</span> <span class="o">==</span> <span class="n">best_xgb_test_size</span><span class="p">:</span>
        <span class="n">X_train_for_shap</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'X_train'</span><span class="p">]</span>
        <span class="n">y_train_for_shap</span> <span class="o">=</span> <span class="n">split_data</span><span class="p">[</span><span class="s1">'y_train'</span><span class="p">]</span>
        <span class="k">break</span>

<span class="k">if</span> <span class="n">X_train_for_shap</span> <span class="ow">is</span> <span class="kc">None</span><span class="p">:</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Error: Could not find training data for test size </span><span class="si">{</span><span class="n">best_xgb_test_size</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
<span class="k">else</span><span class="p">:</span>
    <span class="c1"># Re-instantiate and train the best XGBoost model</span>
    <span class="c1"># Ensure use_label_encoder=False and eval_metric='logloss' are consistent</span>
    <span class="n">best_xgb_model_for_shap</span> <span class="o">=</span> <span class="n">XGBClassifier</span><span class="p">(</span>
        <span class="n">random_state</span><span class="o">=</span><span class="mi">42</span><span class="p">,</span>
        <span class="n">eval_metric</span><span class="o">=</span><span class="s1">'logloss'</span><span class="p">,</span>
        <span class="n">use_label_encoder</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span>
        <span class="o">**</span><span class="n">best_xgb_params</span>
    <span class="p">)</span>
    <span class="n">best_xgb_model_for_shap</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train_for_shap</span><span class="p">,</span> <span class="n">y_train_for_shap</span><span class="p">)</span>

    <span class="c1"># Explain the model's predictions using SHAP</span>
    <span class="n">explainer</span> <span class="o">=</span> <span class="n">shap</span><span class="o">.</span><span class="n">TreeExplainer</span><span class="p">(</span><span class="n">best_xgb_model_for_shap</span><span class="p">)</span>
    <span class="c1"># For binary classification, TreeExplainer often returns a single matrix</span>
    <span class="c1"># representing the SHAP values for the positive class directly.</span>
    <span class="c1"># If it returns a list of two arrays, we would use shap_values[1].</span>
    <span class="c1"># Based on the error, it appears to be a single matrix here.</span>
    <span class="n">shap_values</span> <span class="o">=</span> <span class="n">explainer</span><span class="o">.</span><span class="n">shap_values</span><span class="p">(</span><span class="n">X_train_for_shap</span><span class="p">)</span>

    <span class="c1"># Visualize feature importance using SHAP summary plot (bar plot)</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">"SHAP Feature Importance (Bar Plot) for Positive Class (Manipulator='Yes'):"</span><span class="p">)</span>
    <span class="n">shap</span><span class="o">.</span><span class="n">summary_plot</span><span class="p">(</span><span class="n">shap_values</span><span class="p">,</span> <span class="n">X_train_for_shap</span><span class="p">,</span> <span class="n">plot_type</span><span class="o">=</span><span class="s2">"bar"</span><span class="p">,</span> <span class="n">show</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>

    <span class="c1"># Also show a more detailed summary plot (beeswarm)</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">"</span><span class="se">\n</span><span class="s2">SHAP Feature Importance (Beeswarm Plot) for Positive Class (Manipulator='Yes'):"</span><span class="p">)</span>
    <span class="n">shap</span><span class="o">.</span><span class="n">summary_plot</span><span class="p">(</span><span class="n">shap_values</span><span class="p">,</span> <span class="n">X_train_for_shap</span><span class="p">,</span> <span class="n">show</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>SHAP Feature Importance (Bar Plot) for Positive Class (Manipulator='Yes'):
</pre>
</div>
</div>
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="" src="data:image/png;base64,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AQAAAMAUQgUAAAAAUwgVAAAAAEwhVAAAAAAwhVABAAAAwBRCBQAAAABTCBUAAAAATCFUAAAAADDFYhiG4ewibjaWMenOLgEAgHxjDHVzdgkAbnPsqQAAAABgCqECAAAAgCmECgAAAACmECoAAAAAmEKoAAAAAGAKoQIAAACAKYQKAAAAAKYQKgAAAACYQqgAAAAAYEqev2Lz/Pnzat26tVJTUzVy5Ei1bdvWYb/Lly9r0aJFWr9+vfbv368LFy7I399f1apV04MPPqjWrVvLzc22jF9//VXz5s3Trl27dOrUKVksFpUtW1aRkZF6+OGHFRoaKknavn27nnnmGZtlvb29FRISojZt2uixxx6zGxsAAABA/srzO+5Vq1YpLS1N5cqV09KlSx2GisOHD2vgwIE6dOiQ6tWrp549eyogIEBnzpzRjz/+qJEjR2r//v0aOHCgdZkpU6Zo6tSpCggIUKtWrVSxYkVlZGRo//79Wrt2rebNm6f169fLx8fHukzLli0VFRUlwzB0+vRprVixQuPHj1dCQoJeffXVvK4iAAAAgBzIc6iIjY1VRESEmjRporFjx+rIkSMKDg623n758mUNGjRIR48e1ejRo9W8eXOb5Xv27Kl9+/bp119/tRlzypQpioiI0JgxY+Tr62uzzHPPPaepU6fKMAyb9mrVqqlNmzbW37t06aJHHnlES5YsUf/+/VWsWLG8riYAAACAG8jTORXx8fH6448/1LZtW7Vq1Uqurq5aunSpTZ8lS5bo4MGDevLJJ+0CRabw8HB16dJFknTlyhV98sknKlKkiN599127QCFJXl5e+u9//+vwtmt5e3urRo0aMgxDR44cycsqAgAAAMihPIWK2NhYFSlSRA888IACAgLUqFEjrVixQhkZGdY+69evlyR17tw5R2Pu2bNHp0+fVtOmTfNlz0JmmChatKjpsQAAAABkLdehIjU1VatXr1bz5s3l7e0tSWrbtq1OnDihH374wdrv77//lo+Pj80hUdn566+/JElVq1bNbUm6fPmykpKSdPbsWf311196//339fvvvys8PFwVKlTI9XgAAAAAci7X51Rs2LBBycnJateunbWtYcOGKlasmJYuXaqoqChJUkpKigIDA3M87oULFyTphoc2OTJ58mRNnjzZpq1Zs2YaNmxYrscCAAAAkDu5DhWxsbEqVqyYSpYs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AAAA3PTYUwEAAADAFEIFAAAAAFMIFQAAAABMIVQAAAAAMIVQAQAAAMAUQgUAAAAAUwgVAAAAAEwhVAAAAAAwhVABAAAAwBRCBQAAAABTCBUAAAAATCFUAAAAADDFYhiG4ewibjaWMenOLgEFxBjq5uwSAAAAbjvsqQAAAABgCqECAAAAgCmECgAAAACmECoAAAAAmEKoAAAAAGAKoQIAAACAKYQKAAAAAKYQKgAAAACYQqgAAAAAYAqhAgAAAIApboV9h6mpqVq6dKm+++47/fXXX0pOTpa3t7dCQkIUERGhDh06KDQ01Np/2bJlGjlypCRp4MCBeuqpp+zGjI+P15NPPilJateunUaMGGG9rX379vL29ta8efMKdL0AAACAO1WhhoojR45o8ODBOnDggOrWrauuXbuqRIkSunjxov744w8tXbpUs2fP1vLly1WyZEmbZT09PbVs2TKHoWLp0qXy9PRUampqYa0KAAAAgP9TaKHi8uXLGjRokI4cOaIPPvhAzZo1s+uTmpqquXPnymKx2N3WtGlTrVmzRr/88otq1KhhbU9LS9OaNWvUrFkzrV69ukDXAQAAAIC9QjunYsmSJUpISNBTTz3lMFBI/+6NiImJUVBQkN1tjRo1UkBAgJYtW2bTvmnTJp07d07t27cvkLoBAAAAZK/QQsX69eslSZ06dcrT8m5ubmrdurXWrl1rc5jT0qVLddddd+muu+7KjzIBAAAA5FKhhYq///5bPj4+KleunE371atXlZSUZPNz+fJlh2N07NhRycnJ2rBhgyTpxIkT2rZtmzp06FDg9QMAAABwrNBCRUpKinx9fe3aDxw4oOjoaJuf+fPnOxwjLCxMd999t5YuXSpJWr58uXUPBgAAAADnKLRQ4evrq5SUFLv2cuXK6eOPP9bHH3+sQYMG3XCc9u3ba/v27Tp27JiWL1+uJk2aqGjRogVQMQAAAICcKLRQUblyZV24cEFHjx61aff29lZkZKQiIyNVrVq1G47TqlUrubu766233tLhw4c59AkAAABwskILFc2bN5f071WgzPDz81PTpk21bds2lSpVSpGRkflQHQAAAIC8KrTvqejUqZMWLFigWbNm6e67787ysrI50bNnT4WEhKhatWpycSm0XAQAAADAgUILFV5eXvrwww81ePBgvfDCC7r33ntVv359BQYG6sKFC0pI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H6/Lly9r2LBh2S775ZdfavLkyXr22WdVpUoVzZ8/X88++6zmzJmj4ODgQloD3CryOtd+++03bdiwQR06dFDNmjWVlJSkzz//XD169NA333yjYsWKFeJa4FZgZruW6dSpU5o6daqKFy9ewNXiVmdmvp06dUoxMTGqUKGCXn31Vfn4+Gj//v1KS0srpOpxKzEz11566SUlJCRowIABKl26tLZs2aL33ntPrq6u6ty5cyGtwR3KuENMmzbNaNiwoZGUlGRtW7hwoVGvXj0jMTExy+UuX75sNG7c2Jg0aZK1LS0tzWjXrp3x7rvvFmjNuDXlda6dP3/euHLlik3b8ePHjYiICGPWrFkFVi9uXXmda9d67bXXjNdff93o06ePMXDgwAKqFLcDM/Nt+PDhRkxMjJGenl7QZeI2kNe5dvLkSePee+81li5datPep08f45lnnimwevGvO+aciq1bt6pevXry9/e3tj344IPKyMhQXFxclsvt3btXFy5cUHR0tLXN3d1dzZo105YtWwq0Ztya8jrX/Pz85OZmu/OwVKlSKlasmE6ePFlg9eLWlde5lmn37t3atGmT/vvf/xZkmbhN5HW+paSkaN26derSpYtcXV0Lo1Tc4vI619LT0yVJvr6+Nu0+Pj4y+AaFAnfHhIqEhASFhobatPn5+alEiRJKSEjIdjlJdstWrFhRx48f1+XLl/O3UNzy8jrXHDl48KDOnDmjihUr5l+BuG2YmWtXr17V6NGjFRMToxIlShRckbht5HW+xcfH68qVK3Jzc1Pfvn0VGRmpli1bauLEidY3gcC18jrXSpcurfr162v69Onav3+/Lly4oG+//Vbbtm1Tly5dCrZo3FnnVPj5+dm1+/n56fz589ku5+HhIU9PT7vlDMNQcnKyvLy88r1e3LryOteuZxiGxowZo6CgILVs2TI/S8Rtwsxcmz9/vi5duqRu3boVVHm4zeR1vp0+fVqS9NZbb6lTp07q27evfvnlF02ePFkuLi569tlnC6xm3JrMbNs++OADvfzyy3r00UclSa6urnrhhRf0wAMPFEit+J87JlQAt5opU6boxx9/1EcffSRvb29nl4PbyJkzZzR58mSNHDkyV1eJAvIi87CTevXqafDgwZKkiIgIXbx4UbNnz1bv3r35cA75wjAMjRw5UocOHdJbb72lEiVKaNu2bRo7dqz8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uX2cq1atquTkZP3444/WtoyMDI0fP95u7Jw853PmzNGRI0esv6elpWn8+PFydXVVu3btbO43Pj7e5oOp1NRUffzxx3m634KQ1fblnXfesXtttG/fXpI0YcIEm9t+/vlnu6ur5YfsHpOKFSvKzc3Nbs5t3brVbq7de++9Cg4O1vTp022u3Hj+/Pl82y7GxcXJzc1NUVFRN16xOwB7KpBnPj4+mjlzpjp16qS77rpLTz/9tMLCwpSUlKT4+HgtWrRIixcvVtOmTVW9enWFh4dr9OjRunjxou666y798ccfmjx5smrWrOnw06Tc6NKli958800dO3bM5hP5rIwZM0ZbtmxR+/btVbduXfn7++v48eNauHChduzYoWbNmpn6Mpu1a9fq8OHD6tWrV5Z9Hn74YY0YMUJffPGF7rvvvjzflyM9evTQ0qVLNWTIEPn7+9u8CZOUL8/HE088oVdeeUV9+/ZVfHy8ihcvrtWrVzu87O59992nESNGaMSIEapdu7a6dOmismXL6tixY9ZvJE1LS8vRuq1cuVIlSpSwXlf+RrKaG4888ogWLlyo5s2bq3v37rpy5YqWLFmS7eWBu3XrphdffFH9+/dXRkaGw0M73n77bW3ZskWPPvqoHn30UdWvX18eHh46ePCgVq5cqXvvvdfhNdav17BhQ5UuXVpDhgxRQkKCgoODtXv3bs2aNUs1a9bUzz//bO3r5uamMWPGqFu3bqpXr5569eolNzc3zZgxQ4GBgTpw4IDNp38DBw7Ut99+qxdeeEHr169X8+bNVbRoUR06dEjfffeddQ+OM1WuXFkjR47UL7/8onvvvVc7duzQtGnTVK1aNZtLBD/yyCN6+eWX1bp1az300EM6f/685s6daz15+1pdunTRY489ppkzZ+rPP/9Uhw4dVKxYMf3xxx9as2aN9Y3qfffdJxcXF7399ts6e/asfHx8VLFiRUVGRmZZ74QJE9SkSRM1atTIeqnT5cuXa82aNeratWuW34njTHfddZdeeOEFjR49Wo0bN9Zjjz1mvaRsSkqK5syZY33zWa1aNQ0YMECTJk1S8+bN9fDDDysxMVGTJk1SrVq17K7j/+mnn6phw4Zq3Lixunfvrjp16igjI0P79+9XbGysunfvbv1ugtzIzePct29fjR07Vp07d9bAgQPl4eGhBQsWODxM5u6775afn58++eQTFSlSRAEBASpZsqT15GLp37AQGRmpZ555Rn5+fpo7d65++uknvfbaazbH2T/77LP6+uuvFR0drWeeeUZpaWmaNWuWw8Mc8zLX8kPnzp01fvx4tWnTRn379pWHh4e+/fZb7d271+48v/DwcPXt21dTpkxRdHS0OnfurJMnT+rjjz9WnTp1tGPHjnzd4xIYGKiwsDB9/fXXqly5skqVKiUfHx+1b99evr6+6tmzpz7//HM98cQTatq0qf78809Nnz5d99xzj/bs2WMdx9XVVePHj9ejjz6qevXqqU+fPtbviQoMDNShQ4ds7je320XDMLR69Wq1atUqz5e8ve0U8NWlcJPK7jJ2uu5yoJmyuoTozz//bHTr1s0oW7as4e7ubpQsWdK4//77jVGjRhmnT5+29ktISDAeeeQRo0SJEoa3t7dx3333GYsWLTJ9uVLD+PcSiG5ubg4v6+bokrI//PCD8fzzzxsRERFGyZIlDTc3N8Pf39+oX7++MXbsWOPy5cs2/Zs0aWL4+Pg4rMcw/nd5x8zLZT7yyCOGJGPv3r1ZLmMYhlG1alXD39/femnTChUqGOHh4dkukxOpqalG8eLFDUlG7969HfbJzfPhqM0wDCMuLs5o0KCB4enpaQQGBhp9+vQxzp49m+UcWr58udGiRQujWLFihoeHhxEcHGy0atXK+PTTT236ZXVJ2ZSUFMPHx8cYOnRojh+L7ObGlClTjOrVqxuenp5G6dKljT59+hinT5/Osn7DMIx27doZkowqVapkeZ8XLlwwRo0aZdSoUcPw8vIyfH19jWrVqhm9e/c24uLi7NYzq8tJ7tmzx2jZsqUREBBg+Pr6Gk2aNDE2b96c5etj3rx5Rs2aNQ0PDw+jfPnyxogRI4xFixbZXSLXMP69DO2ECROMiIgIo0iRIkaRIkWMsLAwo2vXrsaaNWuyXLfsas+v7UrmJTl37NhhNGvWzChSpIgREBBgPPnkk8bx48dt+qanpxvvvPOOUblyZcPDw8MICQkxXnjhBePXX391eFnKq1evGpMmTTLq1KljeHt7G76+vkbNmjWNESNG2PSbMWOGUb16dcPd3T3b+XCt3bt3Gx07drTO72rVqhnvv/++zSVYs1rnGz1O18vqNZnd5TizusTqlClTjNq1axuenp6Gn5+fER0dbWzevNmu39WrV4233nrLCAkJMTw8PIzw8HBj9uzZWdZy8uRJY+jQoUaVKlUMT09Pw9/f36hRo4bx3HPP2Vz2OreXVc3p42wYhrFixQqjVq1ahoeHh1GmTBnjxRdfNOLj4x0+RitWrDDq1KljeHp6GpKslxC99jKmEyZMMMLCwgwPDw8jLCzM+PDDDx3WOGPGDKNq1aqGu7u7ERoaarz//vvGd9995/ByqLmda1nNn+wut+roMreLFy826tataxQpUsQIDAw0HnvsMePgwYMO+6anpxsjRowwypcvb3h4eBg1a9Y0vvnmG2PIkCGGJOPEiRM3rM8w7Od3VvN127ZtRoMGDYwiRYoYkmzmbXJystGrVy+jePHihre3t9GwYUNjy5YtWd7vwoULrXMgODjYGD58uLF27VqHj1VutosbN240JBnLly93uK53IothFMJZW0AheOaZZ7R27Vr9/vvvNp9S9uzZUxs3bnT4LaG4Oc2YMUMxMTE6cOCAzTfiTpgwQa+++qr1Kj45ldXcuBOMHTtWQ4cO1Q8//KD69es7u5wcCQ0NVWhoqM23dQPOsnHjRjVr1kzTp0/P0Ter30nat2+v9evX6/z58wVyIYabWefOnXX48GH99NNPXP3p/3BOBW4bo0aN0unTpzV9+nRnl4ICcOnSJb333nt64YUXchUopDtjbqSlpdmdr5KSkqKPP/5YgYGBNt9RAgC54egcxL1792rVqlVq3rz5HRcodu3apdjYWI0dO5ZAcQ3OqcBto2TJkjp37pyzy0AB8fb21rFjx/K07J0wN/bv36/WrVvr8ccfV8WKFXXs2DF9+eWXOnDggD799FO773wAgJz68ssvNXPmTLVt21ZBQUGKj4/XlClT5OHhoVGjRjm7vEKXeY4QbBEqAOA2EBQUpPr162vOnDlKTEyUm5ubatasqffee0+PPvqos8sDcAurW7euFi9erIkTJ+rMmTPy8/NT8+bN9cYbb1ivEAZwTgUAAAAAUzinAgAAAIAphAoAAAAAphAqAAAAAJhCqAAAAABgCqECAAAAgCmECgAAAACmECoAAAAAmEKoAAAAAGDK/wf6gPh7VeSfPwAAAABJRU5ErkJggg==
"/>
</div>
</div>
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>
SHAP Feature Importance (Beeswarm Plot) for Positive Class (Manipulator='Yes'):
</pre>
</div>
</div>
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAu0AAAHLCAYAAAB8o6bRAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQABAABJREFUeJzs3XecFPX5wPHPlK3XG3dH79IEFFRUjA3FTtRYYtREE43+YqJRk5ioiSYmmmKJJvYSjbE3FAsiKlUQBEF6OQ6443rfvjPz/f2xx90texxFqj7v14uX7uyU78ztzj7zzPP9jqaUUgghhBBCCCEOWPr+boAQQgghhBCiaxK0CyGEEEIIcYCToF0IIYQQQogDnATtQgghhBBCHOAkaBdCCCGEEOIAJ0G7EEIIIYQQBzgJ2oUQQgghhDjASdAuhBBCCCHEAU6CdiGEEEIIIQ5wErQLIYQQQoiDyh133EF6evoO3ystLUXTNF577bVdWv/uLrc3mfu7AUIIIYQQQuwNxcXFfPbZZwwePHh/N+Vrk6BdCCGEEEJ8I3k8HsaNG7e/m7FHSHmMEEIIIYT4RuqszCUWi/GLX/yC3NxcsrOz+elPf8oLL7yApmmUlpYmLR+JRLjuuuvIycmhuLiYm2++Gcuy9vFeJEjQLoQQQgghDkqWZaX8cxyny2VuueUWHnvsMX7zm9/w8ssv4zgOt9xyS6fz3nrrrei6ziuvvMI111zDvffey5NPPrk3dmWHpDxGCHHQicfjPPPMMwBcccUVuFyu/dwiIYQQX5t2Xuo09cZ2Zw8Gg9s9/6elpXU6vb6+nkceeYTbbruN3/zmNwBMnDiRCRMmsHnz5pT5jzrqKB588EEATjnlFD755BNee+01rrnmmh3tzR4nQbsQQgghhDjo+Hw+Zs6cmTL98ccf54UXXuh0ma+++opIJMI555yTNH3SpElMnz49Zf5TTz016fWwYcP4+OOPv0ard58E7UIIIYQQ4gCg7dLcuq4zduzYlOlTpkzZ7jIVFRUAFBQUJE3v1q1bp/NnZ2cnvXa73UQikV1q554iNe1CCCGEEOIAoHXyb88qLi4GoKamJml6dXX1Ht/WniZBuxBCCCGE+FYYMWIEXq+XyZMnJ01/66239k+DdoGUxwghhBBCiAPAns+sbysvL49rr72WP//5z3i9XkaPHs2rr77KmjVrgETJzYHqwG2ZEEIIIYT4Ftn75TEA99xzD1dffTV33303F1xwAfF4vG3Ix6ysrL2yzT1BU0qp/d0IIYTYFTLkoxBCfANpF6ZOU6/sk01fdtllzJ49mw0bNuyT7e0OKY8RQgghhBDfGjNmzGDOnDmMGTMGx3GYMmUK//vf/7jvvvv2d9O6JEG7EEIIIYQ4AOz9mnaA9PR0pkyZwl//+lfC4TD9+vXjvvvu44Ybbtgn299dErQLIYQQQohvjTFjxjB37tz93YxdJkG7EEIIIYQ4AOybTPvBSoJ2IYQQQghxAJCgvSsy5KMQQgghhBAHOMm0CyGEEEKIA4Bk2rsiQbsQQgghhDgASNDeFSmPEUIIIYQQ4gAnmXYhhBBCCHEAkEx7VyRoF0IIIYQQ+53qJGiXML6dlMcIIYQQQghxgJOgXQghhBBCiAOclMcIIYQQQogDgBTDdEUy7UIIIYQQQhzgJNMuhBBCCCH2O+mI2jUJ2oUQQgghxAFAQvSuSHmMEEIIIYQQBzjJtAshhBBCiAOAZNq7Ipl2cUApLY1SWhrdqXlDMcXM9XHKGp2U9xylmFfmsLIm9T2lFF+U2SzZYn/t9gohhBBiz1BoKf9EO8m0iwNCIGBz7z8qWb8+EbAPGODhppuLSE83Op3/o9VxLnw2QENIYejw65O8/OUsPwAlDQ6nPR9nbb0C4OzBOq9e4MJjalQHHE57Msji8kQwf1w/gylXppHplRODEEIIIQ5ckmkXB4R33mlsC9gB1q+P8s7bjZ3O6ziKn7wUpCGUCMptB+7+KMLCTRYAv5pmtQXsAO+scXhqcSKrfueH0baAHWDWBpt/zNi5zL4QQggh9iatk39iKwnaxQ5ZtmJVlU0wqnY8825avSbCtoUsW4P4UFyxqk4RijmsqnXYUO+wsSG17OXTdXEA5pe1vqdU27/5rYH6/M2pJTEzSm02NSu21NnUNbe/3xBWrOmkvEYIIYQQe57q5J9oJ+Uxokuz1sf54XMBypscMjwa90zy85NjvHts/Y6jeOL5BpZtslEuE91ReGwbHejXz80zXznc8IlDc9hBjzk4DmR5ICfLpKEpkVnH0EHTuG1qDEfTGdNdp3yFBXb7173QlyizGdPD4IuyDoF7loeZLQZ9HrfJDMUYUB3grLEeYoVe7p8TJ2bDsG46b1zq5ZACucYVQgghxP4hUYjYLstWbQE7QEtU8fNXg5TW7bkOnLPmh/jw0wCqNb52dI2YodO7j5uxJ2Vz9TSH5qiCSCJgB2iKQpPuwlSqLWAHiNrwm3cjnNBDJQXsAFNWJgL8O071MLyw9WPvd4HXZOvtt2a/m/JsHy99EeOvMxMBO8CKaoer3ojssX0WQgghRGekPKYrErSL7Vpf67QF7FvZCj5cGU+aFosrguHk+QIhh0jMoT7YdXnJ0hWpwbA33eCGW4r5vEbhWIqUuhnAQeOIaKQtYO9o5obUi4qVNYrqgIPb1FjyyzRm/18a4wa5E+UzTnuAH/C5aHaldn6dVergOKk36pqjiqglN/C+bZRShJriKCV/+2+rlogiEpe/vxB7kowe0zUpjxHbtXJDDAOF3fqlsVv//fS1EI/Pj/L8penM/SLM89NChGOKo4a6ue676Tz8chNL1sSwNSgxTLJ7e3j2B2kc1rP946aU4ldvh3lvcZyhHbZZaxrMTPOx5Ne1FFg2x2tQmu1ngz+1JKcoEsPvswkZyUF2aUhvbSngM8CdeH/AI3ECjRb9czUe/64HPRyDuniiaM6lQ6aHuK5RleuHoJW0zqHdNHS9/eTREFZcPtni3bUOfhfccJTBXSfK1+nbYNOSJt77xzoayiNkF3s47caB9BuTvb+bJfaRxrDDD18M8c4KC58LfjHew91n+vZ3s4QQ3wKSaRedqqi3ufOFFvLsREdOxdYwOBG4frHZ5pzHWnh8SpBQVKEUzFsR45aH6lmyJgaAoWCQZbGpLM73ngkkZaqfWxDj3k8irDFc1OuJj6ENzM/0kxe06GYlLhUMBQMaQmQ6ydlzQ4c0QzGuKYDeMdvpMfiyUWdYLxPcOnjMRDZe0whggNegpF4x6bkwc9fF23u5xB1oiRE39UQAb7YH6KYOD5zpSdr+TdMspqx1UEAwDn+ebfP6Shn3/ZvOijm88YdVNJQn7hA1VkR5845VxMLyt/+2+PWUCG8vt1AKQjG45+MoLy+O7e9mCfENIeUxXZGgXXRq3uoYlgPZStHPdkh3FNt+edbWO2z7U9XQmBq85Dk2JXUOazuMxPJ+a4mNrWl8muZjtt/Lqm5+oppOnpW6joGhSCJr7tHBZ2D7TF7qU8har5tizUFLd0G2BzI9oGn4fAbHD3Kl7pgr8ZHvbCQczbITQbymQYYb0l2QZpJf5ObUwclZ9A/Wp9bsvL9ORpr5pqtY1UK4OfkuTDRoU768ZT+1SOxrH6yKp0x7v5NpQohdJ+UxXZP7+d9yM5dE+NebAcprbcYNc/ObSzLplm1gejRKDZ2ApuEBspzUgDTNBa6oIs12MJVKZOINjW3LfLMsm7PDUf50W4Cjj/Rz5aW5DMjvcL2oaVSbJkU5Or22xAnpOmEd/I6DrzU7n5FlgK4l/rXK8Wps9nogzQ2+5AB9bZXFqCKdpOtSpSAUJ9GjNTVoT/PoBJqiiR6tupYI2r0mjXGN11c7nH9I+7r6Z2tUBJLX0T9n+yeXt1da/GpqnJIGxcSBOo9NctMjM/ma+X/LHW6bbbMlAOcM1Hj0VIM83zfrhOU4indfrGbOhw0AHHNKDmdd0i2p9GhvWTu9ijkPr6WlMkLfY/I56ZahpOV5drwgsHJeI+88sonGOgv8PgzHwRONYSgFGmQV79x6dkQ5itmPrmfpW1tAg1Hn9uDYq/uj7YPjs7/EAnFm3fUVJR9W4M1xM+bawQz7Xh8AKh9dQfldi4nXRci/sD/9/nUsRoZ7r7WlNuDw09fCbJm+hkc/eZ5DK0rRjxoMD18No/vxzOdRqgOp544B+Z0/BG5fmrne4vq3wnxVaTO+n8Gj5/sZUrj/2yWE2HP2WNC+cOFCrrnmmrbXuq6TlpZGQUEBQ4cOZeLEiRx99NFo23QcXLNmDf/5z39YsWIF1dXV+Hw+CgoKOPTQQzn//PMZMmRI27xjx45NWtblclFYWMhxxx3Hj3/8Y7Kzs5Pef+yxx3jiiSd47rnnGDZs2J7a1W+MLbU2v3uyia2J7TnLYvzhmSYe+WUuf5kZJdBathIFanSdTNumubV+XENx1xl+Pno9gtXaEVMnZdAWLKUYFEtkJuMOzJwTxDQ0rjs/m79PjxBvuxZQtGyJk2cp0LRE2YlhYCibAj9kZcGssMJpDV78MYsrhjj8rZJEfYrHTNTMADiK5pY4sxpIjA+5dXooTocNJvGY0C8Tvtpit62D5hiYOhHN4KJ3HJYXaBySm9j+Yd015pQl76y+nftWGxocvvdyjHjrqt9d4/CDV2N8+uP2Ov3FVYrL3rXbLiVeW61wlM3r3/1mXVfP/qCBj96sa3s9/a06snJMjj8zb69ut740yPu3f4Vq/YCWzKzBjjt894HDd7hsQ2WUl/5cQnzrXRjANgzCXg9p4Qhjzikit8eeqWle/GoZC57f1Pb68+c2kl7gYfT5PffI+g9Ec+5Zztop5QAEqyLMvGMpOf3S8TdF2HDt7Lb5ap5bi+YxGPD4d/ZaW378SogPvwyx8fUH6BZuvXvy2Wo468/M//RfXPlyasf5gfk61x699y4kdkZTWHH20wGaW5s3Y73NpGeCrPpNRspvrhAHNvm8dmWPRwQTJ07k2GOPTYyuEAqxceNGPv30U959912OPPJI/vrXv5KRkQHArFmzuPnmm8nOzubMM8+kV69etLS0sGnTJubMmUPv3r2TgnaAwYMHc+mllwLQ3NzM/PnzefHFF5k/fz7/+9//cLk6KYkQnZr9VZRtK1EWr42ztspi3sbkN5Sm0cuy8MbjBDWdbo7N5Ydn8cEL7YGr09l3rZMfjAWLQhx/ZjY4iq15IB3wdTIKi4NiZDf4YlWI45ww1ekeTFtRGIjQ5PeBaSQC7MZwe/16zG5PpDdGEh1RNQ2iVnt7NA1QbfPN+z8/Y/8ZTG1/1AaXga3gnXWKQ45MLP9VrZaoe3doK7v7sBRuOSZ1Fe+uttsC9q1mlDrUhxS5/sT6JrfWx3c0ea3CUQr9G/Sju/Tz5tRp81v2etBeMqumLWDfauO8OqyIjentOhu5+vMmbFu1X/y1UrrO6b8ZzGETC/ZYO9d+Wp0ybd2Mmm900L5hekUn0yopLmtMmV7/xoa9FrRbtmLKCosTt6xtD9i3Kq/nizdXA32SJrsNWHJTBn73/v2OTl8bbwvYt1pT47C80mFEsWTbxcFDymG6tseD9iFDhnDGGWckTfvlL3/Jgw8+yP/+9z9uvfVWHnzwQQD+9a9/4fF4eO655ygsLExaxnEcmpqaUtbfrVu3pPVffPHF3HzzzXz66afMnDmTk08+eU/v0kHj/YURnpoWojnkcPpYL9edlYbLSP4C1AYczn+qhfmbbNxAga6R77R/TXQdyqot0twQ3KZgPc12yHMc0Bx0HT6cG8LQwe6ilNtsLW9RQItpENZ11qBz1iPNWCTiXZ9S5HUI4DuqNg0eqjPI9Cpimkmt143LUeiWQ7HflajR0TWwbIhYqSvY+kg1j5FoaIfAbVAowqBABF1T/PS+CIbHgx0nuXKmQ1lCYZriT585/GeZQ2249b0OFTiLaqH4YQuv5hBtiJHndrj5eC/FGakpeK+mmPRUC+vrFacONjm0f2p5RYZL8cN3baZvVBSlJbZVGYTje4A3FGPamjg9s3T+eJqX7wwwuXWGw6urHQp8cNsxBucekrrdx790eGChTcyGKw7VuWwo3Px+nNmlNocW6dz1HYPS96tYs6iZ3CIPEy8tov/w9E7+MjDnkyamv9tALKYYOtxH86YANeUxjCw3wShk5Zicfn4edVVx5kxtQAFeT+oJOTM39TS0uMzi11PCLK+0+U5/k59lh1j9zhaiQZvhJ+dz9A+KO23TppIIb/2vmsryGAOH+jj/8kKyckzi0dQPqS/LRe3GILMe3UBtSZCeo7Loc0QOS97cQqgxzuATCzju6n5k5HaeCFDA2x+2EEv3cNSxmZ3O05W6jSE+eaSEqtUBiodlcOL/9SfsdlOTm42j63gjUTIDQdLyvl4WNx51mPZMGSvmNJCe4+KES7ozZFz2zq/go6Vw+8uwuQ7OOhz+dilk+r9Wm5i1Am59AdZX4i86gxjJx7j+y1oGrVzOQDYTw0UNeYTxkZvZAsf8FjbWwJljKD/iBEoeXocdjNP9kv4Mun0kmrF7XbUMHQrSNSrSstumOWjcNe4s/jN8PLGm3JRlMr0axz4UoD7scNEoN3ed7sVtJn/Gq1scbnonzPS1FkO6Gdx9hpej+nT4zL82F/78OuUhjRvPuoJZGT0ZUWzw1zO9DC8yuP2DCC99GSPXp/Pbkz1cODr189A9S8e0Le78bDKXrJpHk9vHP444nYJTP4VYEH56Ktx4DuX3LaXy0ZWgaRRfNYjutcvgxVmQmwG3nAsXjd+tYyeE2Df2yb13wzD45S9/yfLly5k7dy5ffvklo0ePZvPmzQwYMCAlYIdEeU1OTs5Orf/II4/k008/ZfPmzXu66QeNL9bF+M2zzW315E9PCwHwy0nJAdfER1tYtCWRHo4CAZeJJxYns3W5mAO/e6qZ605K46+fRNuWy7VsesUt4ppG3NCIOxrPvNGSdE1sOApbT2TlAVAKv2NjAWFDJ2AaBDWNJYYL1Zjo2KoAQyn8KjF6TKJGOLG8BZSZBpalUe/z4HgTP+wRYEluOqvqOozy4m7NsofjgAZpZiKI1zRIb/2R85qJUhqgTyjK2Kb2zHpGbZCGLFjr98LWjL+htT58CVyOw4YmnT/M7RDRb3NB1BRL/AMdTA8VVUF+9HKId670MzwXlte3z3tIfZD6GocKv5dnF8Y5qUmR43LTEG9fZ2NQ8fzyxPYqQu3H5aX5kbYLlPImm7OfCnLh+DSeX5FYbnMzfO8tm/mXw9ji9gDmjdUOP53anvK/bZbDIzNjlDe1bqPFJvfTMno3JT47zfUWT99Rwq8fHUpmXnJQtfSLAP99rKrt9azqOJ54HDSwwom2NTVYPP6PcjS7/aLQtKzElWGHz0hRXvLlWktEccpjAeqCiXZ9PruRkevbS0bmvlCOUqlBeDhk868/byIYSLz3xdwW6qrj/PyWHnz+xhYcTUsaaWjM5X15/aavCDcmPhOrP65h9cc1be9/8XIZjqU4/roBFBS5qKmM4XQYXtTWoHyLxVP/qiQ7x+SQYTsfyNqWw6u/+ormqsR3bN3sOqrWBiiJ+1CtdwuDaX50l87YH/TpalU79MHjm1nwXmK/mmvjvPSndVzz0DCK+u9Ee0uq4Mx7oLXEjcc+gsYQvHTD7jeooh5O+xOEEvveN7CBL3sOavtMuGMW/SfPwRVqBMCFjZctbNYK6V82DzYk2lLz5BKW/qf9/Lb+L1+hmTqDbhu5W83SNI07MzZyTX5vXjrkSC5e/Tl/O+J0/nDMuYkZwqnL1AYVtcHEd+rvn0axFdx7TnKp1IX/DTJjfWKeimaLUx8PsOHWTHL9OsxbDRfeC0rx3UtuZyE9oEVR0WKxqCzIDw538eDsRPZkU4PNxc+H6Jmlc0y/5J/ucX1M/rvmHS7+/N22af99/4n2GW76D01Lmyl9tn0nnF89D2xIvNhUC9+/H3rmwbEdB+EVQhxI9unoMZMmTQJg9uxEnWLPnj0pKSlhyZIlX2u9ZWVlAGRm7nq265vi/S+iKR1A312QfL+0JuC0BuztlKZRZ+hYQIzEaC6ROIzO1PhFP4dDIlHGhsIcHwi1DsGosNDaqkI6cilFVtzCZ9n4LJusuIXXUdS7TQJmItipNfT2oL5VS+trR9OIahqOUvQv1Fhv6lhb33Nvk4fXNKLbPtjEpSdapWvgMhLBur9DsOkyIMNNbrbB0WaUbfUJRUHTOLafkei0SmvH1aYI8ZoQzy7ehWH9dC1RrgO8uiTOT/xBjqhrZmhTkBOrGhjSEqJPtP1WxsdrLezaEDRHIRCDhkhirEloLb9pD3K3vaMQs+H11cnHwlHw8srkaS+s2CbQtZ22gB3Abdv0ag3Yt4rHFMvnp97xWjAndbSUuGFgbVPYrxSoDn9ut2WRHoniicXxxOKkR6KUfZVcMvPh6nhbwA4wqiG1pGbFp/Wp05YE2wL2rUrXRVj0YQ3RoEPc7SJumliGQcztwsxwtwXs27NyWhWmS6d7Xy9uy8IVj2NaFu54HE+8/e/w+dxdGz2mbGlzW8C+VUtVFGObejUrx0/BwM7vdOyspdscK8eBr2amHr9OvTavPWDvatqueOvztoAdwNvsYnhJFd1rmuhd0cCo9Vvo3hqwb2WgGHSKG63DMa/QU0uGKl7asPvtAn469U0+e/EuluX14MPew/jfkHG7tPwLi5JvT25pctoC9q2aIzBlRet+vDQblGJNdiELi/olzVcXUvz3i+T1KQUvfdn58JIXrZzfZduMtz9Lel3AluQZlIIXZyPE/iSjx3RtnwbtgwYNAmDjxo0AXH311cRiMX784x9z8cUX85e//IXJkyezZcuW7a7DsiwaGxtpbGxk8+bNvPLKK7z66qv4/X5OOOGEfbEbX0t9fT3RaPsPViAQoKWl/Qc/FotRV1eXtExFRUWXrysrK0lPffYQGa0jj2zdhs+lbZscBsBEw9a0pGDajjWTpysOjcToG7PaSlcMpciLxsiNxtj2KkHRWpvuOPgcB6N1WpVpEm0tMzE6e4CgrlGan0a9302t22RVupfpjpd4x9vcO/HgQR0Sga7RWtee1DjVWhqjoXtMLE/qTaZYaxtL6uzEOtLcibKa1u1nb9vXcEdPw2wtDfJqMTK8iv7BCCObgnSLJgLFeIdAXEMlLkJC8cTdgLjdvv5tN9PJ39DXyYH1ae2fs1gshpdtLlS2uXiyNa29TR140xLHoONn15eWeurY7oi6quP/JrLdHsvCY1noSrWtf+s2fGZyQBjppNyh4zJVVYmMv9+fWmCl6+DQekdF03BMA9tlorkN0nJ23P/Fk27SVB1ly7rEd9RwHHTHwWn9vnjiib+lUuGkp6Pu6HtueDv/7Gx7QetPb/+cVlZW7tI2tp5LOh6rrbZO6+xc0nEbwU4Okcrwtl1Q7s75qpHkCyUDi7RInJ7VzRTVB9AdhdPJJ8k1IPnOq6lSL7jMbHen+7HT590sP+MqSrhr7pucumkFOdFO+rl0IavDeTgQCGBHA3TygGWyvYn9a9ETF5mZsTB6JyN0+V2p09I7TOu4H1pW13dOHF/yj4RNJ3/c7LS99hu1W38P2cY+2YY4eOzToD0tLQ2AYDBxIpwwYQJPPPEEJ598MlVVVbzxxhv86U9/4pxzzuHGG2+koaEhZR3z5s1jwoQJTJgwgXPPPZe//e1vDBgwgIcffpjc3NSawwNNbm4uHk97/XJ6enpbx1wAt9tNXl5yp7zi4uIuXxcVFXHhcX6y/Mk/dD8+NS1pG+kejR+OST5Ru5XigpHJAWyfbgbde+by7GpFx/DJcBzSLBuP4+CzbfzbZAW1rcM+dlDlMmk0DcpMEwcotG3c2wb7aW4ibpNqn4saj4uYrtMUActl4rQGMUZkmx/oTk44TtxJZLg1LVHiEragJZoI1jv89tWGYborHavD4XKA1ek+UIqK5tYJgXgie5/t5dD+Hv54vKtjiTu65Ww/cI/bELPJ9MKNJ2Vy+sQcfB2Gb1TASp+nbXmlEv1k214o8Nkdjv7W7XQs92nVJ0fj98clRwbd/HDtEe1XGW63m18d60+68WAYGhMGdSj30HWq+mcnraegh4cR47KA5M/uCROz8XiTP29uy8K1zWciLcPA7HClaHld6B1qfg1T49jz2svjcnNzOW2ojyN7t7fr87wc7I53WjQ48sL278DW8rpDDvXTu39yYHL0iVkcM2kg3QamJU0fdXYxA47Np3h48t05w518Sjzykl7895ZVNFUlSq0cXcc2TJSmo6GRHYqQYdicfk73pFE6dvQ97zksh35HJgehfY/KwZeb3LfhxPPaO7kWFRXt0ja2nkuOu7AoaZ3pOS4Om5APdH4u6biNtB+fAv26Jc2j/eqctqGSdud8lf2jU2FIj7bXhZQmvR81TNZnJZdM6ocW4bprEvRvn97H2YDZ8XBp0P+m4Z3ux06fd288J1Fq1+rXCz5ICqa9dtd3Zn57cvt3Lj09nV7dMrlmm5FlRhbrnD40sY2MG8+F/EyKQs38aMWcpPmO6m1w1xnJn9s8v8a1x7YH50n7cct5nXb8T+ygiXHbeWgdPt/lxoDkeXLT4epT9tpv1G79PWQb+2QbBxZ5uFJX9ul4cluD9a3BO8Do0aMZPXo0Sik2bdrEwoULee2115g5cya33347//rXv5LWMWLECK699lqUUlRWVvLCCy9QXV2NaX6zhsbbVd1zDV7+TS6vzg63dUQ9YlBqh6Unv5/G6B4RnpsXJcut8YczfBx3iJtpiyLMXRmjZ57B+eN93PFWkLCm82Waj6K4haEUPTQLgu0/YNnxOB7HIWromI4izbKodbsJ6RohQ6fJ0Klp/bsoTaPMNMhyFH3jFqF0E2++m3VBo62MJCU7rmlYHhOX4/DwJA//mG+ztoFExGs7oMBvaKQrB8dR1G4zcpAWt1GOStS5dxzHXdNo8LmZ2q+QAY1BdKDvIA9Vpaq9nh0SmfKoDV6T9BwXZw7QmXuJxnPLHZSjeGR+azsMLXFeUXBCX50ReRBussk91MPV49wMzDcAg7vv7M70T1uIRBSjDvdzWLXGg7MiVDQnB/69s3UuP8LN1ePczKrQmF7q0CcrcQw3NcMpfb34LZN3Vlj0yta55mg33TJ0hhc4vLpKUeCHa0brFKcnn+xGddP44ocmTy5xiNjwwxEahxeaPLfIYlapzcginauO7MX6BZmsWdRCXrGboybm4fKkXtt37+nhd3f3Yfb0JmIxh9Fj06laH6SmLIo320VzSJGVYzJ+QjahFpvPPm4EBeNOysZE8cWHtShHcfgp+RQPSM4Q6rrG9GszeGJelK8qbI4f4Oes7iNZ+l41kaDF8JMKKB7q47P1pCx3/e97MWd6ExVlUQYN9XPEcZnousb3HxzFkncqqNsYovfh2Qw7uRuarnHB/SNZ+k4FteuD9BydRa/DcvhqSgWhhhiDTyxAuQzqy8vaPxJa6rE4tJ9JQeGudxY9967hfPVeJZWrWygelsGhpxdxapPNvA/rCDTZjDwmi8Gjvl5pDMCRZ3Ujt9jD8tmJjqhHnFFA+k7cZQASHU7n/RkemwYba+HsMTDpiK/XIJ8H5t4Nj06FdZV0O+0wPBk9qfnvahqro1iHdqP3Rf3wl5RhfbQO/ZACPD89Ci3LB/Pugcc+hNJq0s4ayzGHDmPzk2uxA3G6X9KPnKO77Xj7XTn6EFj4d3jqI1iwnrPmLmHOy3/hv0OPwW9FybRi/H7cpKRFNOBHR7i4bKybEwemHtd/ftfHuD5ma0dUnavHedoHCOiZD1/8HR77kMfrtvCdUXXMcBdzaLHOVeM8pHs0+uTovPxlnFy/xk/HeeiVs51c28XjoXtOosQlOw2OGwoffpl474qTSBvVj1HHDKHq6dWgaRReeR7UVyRKdHLTE51Ve++5kZCE2B1SDtM1Te2heyRbx2m//vrrueyyyzqd55133uHOO+/kiiuu4Gc/+9l21xWNRrnwwgspLy/n3XffbcukjR07lvHjx/PAAw+0zdvU1MTFF1+M2+3m5Zdfxuttz7TJOO2776b/NTNtThif4xDVNeoMg2OJ4G1oradUnY/2UuX10GIYrPV2HcQMKjQYfFgGT37ZIQUeiLWVlLQxNFxug8bfejj1qQhzOg5FqRS+eCJzr2vQ4PckZZo0pRJBu0ai1GWrbR5UY2qKa4c7PPRpap073kQd/Il9DT6+qH2P60KKwvtiKePSv/Y9k/OH7vwQa2c+0cJ7K5PLQc491MUbV3z9YO2bLB6P88wzzwBwxRVX7LWhXjctb+E/N65oe23pOmqbwP3Q43O48Df998r2xX70wDvwy2eSJj075kR+dHzy75vbgPo/ZZHWychIQohdE9auS5nmU//qZM5vp31aHjN58mQAjj322C7n83g8DB48GICampou583KyuLaa6+lvLycF154Yc80VBCviFFg26QrRZ7t0Cce56STMmgbPKP1AUgdRXUdS9fxKYV/m3Egt32g409O8HLNYTpJ/Uu37WwKoOvEYza3To3xi2M7BGatNephXafJMGjQDbSOmXpNQ+l6Yhx3TWt9CmqHZTuwQhYPzY6njAYDJLL/jVGuHZU8Oc+v8YNDk78+A3LgrEG79pX6+Xhv0h1tXYPrxu+Zp2uKr6/XsHS6D26/M6hvc1GpG3DUWV8zuysOTJccB/nJ5VPnTexGz6xtyhCPckvALsQeI+UxXdknNSW2bfPQQw/x5ZdfcuyxxzJ69GgA5s6d2+lTUhsaGli6dCmGYdCrV68drv+MM87gqaee4vnnn+fCCy8kPV2ylF9HVb3F8rXJIxR4FIwa5OXEW3vw8fQmbEtx1Lh0yjbFKC2J0HeABzvNxawlUWaX2Inh9UhUsmjAvd9PZ35JnOaw4twxHs4cnQhMZ19m8q8vHEJxxSEmvPBRlC0eF1FTT0SwtgO2w2PzY/x2nI8JtQ2s8/sIaxpV22RXleOAchKRVEe6nigY924d091J/NO1xBNSt47G4jVAU4n32s4XGjgKPe7ANvcWnjzL5LAim+kbFEPyNW4aZ+Axd+0Ec9pQFx9dk84T86LomsZPj3bznQHygLADhaZpXHr3ED57vZLKdUF6Dk0nv7+fpZ80oJsaR55RQJ/tjGMvDnLdshPlOPe9DWV1MOlIMq44ic+aFPfOiLKu1ua0Q1xcc8z+fRqqEN8kUh7TtT0etK9atYr33nsPIOmJqBUVFYwbN44///nPbfP+5je/ITc3l/Hjx9OvXz9M06S8vJz33nuPuro6rrrqKrKysna8E6bJFVdcwV133cVLL73ET37ykz29W98qnQxi0DZ90GAvgwa3lyCN2aa8tc/gOG/cnxgisGOIO7S7wUXjUoe4OaK7zh998O/5FnNKHHQNCmJxynR3W6mM33YYGIry9EtRCqMWmVaQdT5vStAOGnRScwwk6ua3jgQTt8lvCDE4HEEBa/xe6lyuRIDu0jrtXNrZA6RchsYNR5nccFTnm9xZJw1ycdIgCdQPVN50kxN/mDy84NBxO/cMCXGQG1AE/746aVLPbI37J207lJQQQux9ezxonzp1KlOnTkXXdXw+H4WFhRx++OFMnDiRY45Jfsb7H/7wB+bMmcOCBQt47733CIVCZGVlMWTIEG688cZderrpWWedxZNPPsn//vc/Lr74Ysm2fw3F+SaHD3WzaGV7tr0oz2DssB2XbRze12RId4NVW9pLVUb0NBjZu/OgtKLFYewjYWq3Dg+e4aN7KIqhFLam4bUdjm8I4FGK+Ysh4PXQ4jJxAy5HEe9Yd2MkxmjP8UJDxwehKAXh1qjba1JkWZxW19hWGzY4FOH9gmyqtdY26lpSbX33TI2zhn67OzoLIYQQYv/aYx1RxTdLIOzw/LstLF0bo18PF5efmU5h3s4FrtXNDg9+EGLpJovRfU2un+gnL6PzDPhfPo1x60fJw6hlGorjPTG2uEyyGqNkVrQ/JKrK42obBjKiaZS7TPzZBumZJj6/zhmDTX5yuMH3X4kxd3NihJmtmXOfGw4d4GHMymqipckPETJ7+/i0KJc19a3zW04icNdh3jV+juq9851Lxd63rzqiCiGE2HeC2vUp09LUP/dDSw5Mkj4UnUr36VzzvR2XJnWmW6bOXRfu3J2Olk4e7qe7NN7+XWLM/edebeTNDkF7xyoVr1IMiMX58Vg3V52VPJ7xSf105m5MHkLS1GD+pSZ//7Ni1TbbHJKtccoENxe8Emstk2kP0rtlSI2dEEIIIfavfTp6jBDb+v5Ic+vDFdtcNrr9WvK4o/ztI9ZAyqg0pgETxqTWyl/cxXqPHp+RMv/R4zM4baBO/jYPFTy2t06/7Y2LLIQQQog9SEaP6Ypk2sV+NbJI5+1LPdz1aZyqgOKCESZ3nNRe6tCvt5vf/jyf16Y009TscM5hXpoMk9lfRcnP0rni9DT6Fad+jIcX6rxzWWK9lS2K84eb/HFCYr3jj88kGnWYMb0ZTYMTJ2Qx7thEID/9hx5+Nz3O8mrF8X11/naKlF0IIYQQ+4KMHtM1qWkXQhx0pKZdCCG+eQLaL1Ompav790NLDkySaRdCCCGEEPudZNq7JsW6QgghhBBCHOAkaBdCCCGEEOIAJ+UxQgghhBBiv5PymK5J0C6EEEIIIQ4AErR3RcpjhBBCCCGEOMBJpl0IIYQQQux3Uh7TNQnahRBCCCHEfidBe9ekPEYIIYQQQogDnGTahRBCCCHEAUAy7V2RoF0IIYQQQux3an834AAn5TFCCCGEEEIc4CTTLoQQQggh9jvpiNo1CdqFaKWUonxlAN3Q6H5I+v5ujhBCCPEtI0F7VyRoFwII1Md44ZZVVG8IAdBjSDoX/2UIvgz5igghhBBi/5OadiGAmf8tawvYAcpWBZj9Yvl+bJEQQgjx7aLQUv6JdhK0CwFUrAkCiZ7rjX4fW3KyeWN6mKce3EIs6uzfxgkhhBDfAhK0d02CdiGgrYY95HET9HpB0wCNhXNaeOfV2v3bOCGEEEJ860nBrhDAdy7ryaaVLaxq8pJhJ48Uu3xxkP6Nq1g/pQx3lotRVw+mz8nFKetY9GoZy6ZsQTd0Dr+wJ8NOK8KK2Hzxz5Vsml6Bv9DLYdcNoftRBftqt4QQQgjxDSFBuxBAWo6LDecOYf0btYxuaEl6zwxE+eKfK9tef/R/8zj75ePpNjq3bdqXb5bzyQNr216//6eVuNNMKj7YxOpXNgLQtCFA9aK5nPfeyWT1kdFphBBCiI6kHKZrUh4jRKv/LrH5KieDgKGjKYWmHHTl4F9a1jaPpeus6V7EvU/VM3laC7F4Iiu/4r3KlPXN+McK1ry+KWmaHXMoeVc6uO6MwLxKSq/6lNKrPyW4sHp/N0cIIYTYryTTLkSrNLfGFpdJabqPUY3t2fZN3Qo4pD6A4SjmjxhEXXYGhGDlS40sXRnh9l8U4PKlXv+GNgXxWU5rfXw7l8/Y6/tysGv+uIw1p06B1lKluv+sZvDH55AxPrUsSQghxDeDZNq7JkG7+NZZtjxM6cYYgwZ66NPbzYLPA0Qjiqv7GTy/NsCIpuTymJjHTVN2BlrcTgTsHSxYEmHaM5sZNrGIzYsbUbZCtx00pfAGohi2g222B+nuDJOBk3rtk/080ETXNhCYsgGzRzqZ3x2A5k4cl5bFddS8uA5VHybvtB74TuzFit8uJORykebEMZTCiTts+s08ss8fgN4jnabqMK6NBvFB9j7fD6shSsNr61G2Iud7/XHl+/Z5G4QQ4ptJgvauSNAuvlWeeLqGT2YEEi+UIsurEQ45ba+PtCx0lbpcMCMNPRLrdJ2fPbeRbCfO+Kv68fmDq8FJrCCa7iGk+wn7vPiCYWzTIOfQHLy5nr2xawe05tfWUnbxe+2Z8yMK6TvjAsofW826X37eNt/mZ9bRWJiOZQOZaTQ6DoUNLWg2NM6tpXFuLQpoSfOQ7fUTGRGHK/fdfkTWN7HqmDexqsMAlN86nyGzvotvWO4OlhRCCCG+HqlpFwe9hhaHL9bEaGkNvgM1UcoWNdBcG6VkWYCWhjgA5VtibQG7NxbDE7faA3YATcPWdQwrOXurOQ4u5WC6DTxxK+m9tEgUT9wipAyWvbqpLWDfuj7H0All+KkryqMxP5sNFYrKkhBdaSlpoebzGqIRmwUlcTZUt7fHjthUz6shWBZMWqa8RTFjsyIYU9Qva6BuSX2n61ZKUf9lPQ3LGrpsw+IqxcLKTq5edkFkTQOBORXEojYbb5rVFrADRBZU0fjcSjb8fnHH1tHi99Dx8Du6TmOaD6fDqUoD0kNRcBy8y1w0LKpL3faqeoJzK1B252PsO8E4wRllxMsD7Vt3FE3zawh81fmxA6j86+K2gB3Aro9Sct1sQpuD213mmyRe2kR45maciLXjmXcg9kUFsUUVba/VqgrU3LXb/Zt15Gxpwp6xDhWIprxnVQYJzdiM09L5RfbXYZc1EZtRigqlrttuihKasRmr6tvxWRBib1Cd/BPtdjvT3tzczOmnn040GuXOO+/kzDPP7HS+SCTCG2+8wccff0xJSQnBYJCsrCyGDBnCKaecwumnn45pJjdjxYoVvPLKKyxevJja2lo0TaN79+4cddRRnH/++fTt2xeAhQsXcs011yQt6/P56N27N2eccQYXXXRRyrrFN8srn4R46M0W4hZ43fC9HlHsyesIGS6aM9NRmoZhaky8rJiM/ulojkP/qloywxHK8nNpSvMnrc+0LA5ZW0pFUT4tGWl4IjGyGluIpPtIC0cYs3YD64u70ez3kRkOkxuOUlOQl1i4roFtx4TRndTyjSfvL+eXd/cnLT25tt2xHeb/cgGlb2ykMjONZ48bTYMnkZX/7lgPtwyKMPeauUTro2i6xqAfDWTsnYdx+2ybu+crbAVplsXPpixl1MYqcg7N4cTnjsOb7wUgUhth5vdn0LisEYD8I/MZ/9/jcGe629rQGFGc9abNnNa+socXwvvnGXRL2/lblspyKL1sGo0vJUbTqc5Jo3tjS8p89W+UYLfEty6FjkIZqduJGSZRbEwcTBIBna7AG7CJ+SG0MQhHta4lbrPx+1Npen09AO6+mfR7/2y8Q9oz4S0flLL54vdxmmJgaBT89giyrj6UJadNI7gicWxyTunOyLdOxvAnnz+iG1L3o3FmJR+Onkz/qw9h5N1jdvo4HWxqfjGdpn8tAgVGgZ+iNybhG99zl9dj14aoOeMlYgsSAbv7qO7kdwugv7MoMUO/Anj/l2iHdN5/IXbn+8T/9CHYDmR68bxwOeaZwwGou/tzan8/FywHPcNN0bMTyTh30O7t8DZafjuN4N/ngu2g5fjIfvkCPKcMAKD55dVU/PhDVDAOLp1u9xxH7o3f3M+CEHuL1LR3bbcz7e+//z6xWIwePXrw9ttvdzrP5s2bueSSS7jvvvtwu9386Ec/4tZbb+WSSy7BsizuvPNO/v3vfyct8/jjj/PDH/6QOXPm8J3vfIebbrqJG264gbFjx/Lhhx9y0UUXEQwmZzImTpzIH//4R+68806uuuoqLMvi/vvv569//evu7p44CFQ32vzz9UTADhCJwUvrTMJKozndj2rtAGpbivf/s4Ucr6I4HCQzHAEgvfW/HWU3BzAch55bqhm8ppSc+mY29+5OQ2YG/lCEtFickRvLGb9yHf1q6rE97QFv0OdNWV/LNhcFCthc7fDB2/U4tsKKJIJ6O+awcfImSt/YSNTQeWvU4LaAHeCthVEev6eEpmYrkX1wFGueXsuHU2u5a55qS2IHTZNnvzMSW9No+KqBL/+5gqiVeHP5P5bRuKyxLXNR+3ktax5dndS+vy1w2gJ2gEVV8PvZNnaHOwi2o4jEU/MfynZwIhYNL61tC9gBujUEaW7dl46Zk5oZtdh64m+kt56q3bHUixxXzAE0LHTs1rUk/qthRhyyD8tum7fu6ZVtATtArLSZLb+c1d5Gy6Hsyo+wm1ozpbai5q7PWXv1TIIrGtraWD9tC+UPtw/zuVXW6b2T9xkImyYoKHlsNdWfdMgcOwortGsZaWU7qHBqFlcphROKd7LEnqHiNira3lYnGEOp9r9x6OONND20qO2PZ9eEqL76Q1TMRnXyN9vudkIxmu+a3RawgyI2v4zmd0ppW/mGGtQvX+x0eWfZFuJ3vA926zabI0SvfAkVt4mtbaD21tlgOWjYOC1Rqn4yFac5ApadOEHsKqUgFCU2eyPBe2YlLhQA1RCm6SeTE5/56hYqr56WCNgB4hbVN39KfGNz+2qaQqhw6l2BfSIaB2vf9/0QQux5u52Gnjx5MmPHjuX444/n3nvvpaysjJ4927MukUiEG264gfLycv72t79x0kknJS3/ox/9iOXLl7NixYqkdT7++OOMHTuWf/zjH6SnJ+ctf/GLX/DEE08k/ZgADBkyhDPOOKPt9QUXXMD3vvc93nrrLf7v//6PnJyc3d1NcQBbvcmi4510w3E4tLoOy+clLRojZprEXCZoGkrBmz+YS48cL1tDk6xgiIjLRX1GGkrTyG1spveW9qEFDaV4fvgAvlPfjC8aTbn+j7vdSa/Dfh8NmelkNwfQlAKliOg6MUPHazvEdZ2a9DTQNZbObaT6uZVEmmIU+RVWeQuVbi9PnnQEawpy6BWLp2zvP/36U3rcGPKbg/xg5lfk2vD4uzFojSP7V9Zz5ceL6V3XjK1rfDq0Dz/2D8N+0OaSoRrnzqslbmqJ0WyUwrAV9YuTS0E+ryDFY0sVLyyN8tvxLlyOw19mWTRF4OxDdJ6e5CbXr1H5t8VU3rMYuymGp286iuTuREbEIY6O1hqYWRjEomC5QO/wN/SFLWxDJ+xLnJrcEQdvcOtfTEM3HXKsMBYaOBDEy+LBk/EMyMDqlY4+dzPJXYWhaVoZZW9tonhid7764Swqq1xoejZpKkqWSpS62B+WkIei3uujIT0NpWsse2ItBZcOxFfU3tG08BeHUj+9nMB7GwmYbiK6C4WGEXOwXRoLT/+Iou8Ukn1OL9Y+uIJoVYT84woZ/cjR+HulpR7cDuIPzSD+xw+gLoR+2lA8//kBWrcMGl5aQ/nNc4iXB0kbX0yf/0zAMyCry3XtLKUUsd+9S/yhWRCz0U4dRrg8ivVlJUb/HDIfOgPPGYOJfp46pGl8ZR0lWf8ENDKuGEHBP09Cc3U+MpJaVYl1xbOoeRvwut3ESMdHA+nUoOGg0BJ3WjBQ+OHzDZ2v50+v4WcL4BDHTw0DiFf7cA1/Eu8lIzBUjGLW4KMxsZ56E5W7CMdw0KwY2nlHwJM/hSx/p+tP8soc7BueI1DhJU46BgoHA9Wa63I2NeEMu5H4mjocDgMcCikhiypQGtb/gbrvYuwJf8cpS5TE6cf1x5h6I5rP3cWG95BAGK56BF77DHxu+OXZcOfFe3+7QnwNkmnv2m5l2letWsWaNWs488wzOe200zAMIyXb/tZbb7Fx40YuvfTSlIB9q+HDh3PBBRcAEI/Hefjhh/H7/dx9990pATuA1+vl5z//eafvdeTz+RgxYgRKKcrKyrqcVxy8Bvc0MTp8gg+pbaBnSwCNRMDosSzMrRk5pUhvCeKqaa9h1oCixia6NTTiikQYXFqG4bRHkNV+L+vS/XzaLQe3nlpb54qlZu40x8GMxTHiFqZl07OmnpiCtQV5lOTn0uLzEnK5iJY0EWmK4wlGiZU04UQdHhkznDUFiQvMqJ564qrOSQR9tZlp/Ov0I1ndr5jcQKINhu1ww7vz6F2XyO4ZSnHkunJQirgDz33lUFMVbR9+UtOwDY3MQzKTtjG2qJMDrWm0aCa/m6P41XSbhnCidH/yKodfvB+n6YNNlP9mHnZDFBxFtKQl5cTbmOltnZL465g4eEwbd9xJmlMDMgIx8mtCZNdESGvpePGiyLQS2UoTRZ4dxO0kspsNlREavmwgaqTmISK6wefXzGXVrxdS+cbmxEWcphHQvQQ1Nw4amqOImCZ1memo1mMfaozxxY2fJ61LM3X6PHo8W7IyiRhu0DQ0wLAUuqXQbUXt9AqW3bqIaFXiTk7trCq+vPazTg5sO3v2euK/eB1qg6AUzvsriF3zMtH1TZReOo14eeLuYnB2BaXfn9rlunaF9ewC4vdMh2AMFbcJvLsB68tEgG6XNNDwvVdwaoOYfTI7XV5FbFTEovmRL2m8f+H2t3PhE6h5iUDciMXIpp4MatBJ/P311m+Xho1GGMb0SV3JtC8xXvkUrXUZNyGy2AJAfG0DwYcWUMg60mhEw42Gmfi02Q7EAEeD1+bDzc/v+MBsqIJL7m8L2BNt0zBw2Hom0HULfc0m3ITRscijnBwq0VHoOLjfm4M65nacsjBbP/fOrA3Y13d+F2GP++3z8NLsRJa9JQx/fAVenr1vti3EbtM6+Se22q2gffLkyfj9fk4++WSys7M57rjjePfdd3E6BDwff/wxAOeee+5OrXPJkiXU1dVxwgkn7JHM+NZgPTOz8x8bcfArzDX42bnpbdnbwmBqB0/TdkApCqrqcccs0puDScF3xGUS1XQO2bCFuK63vRcyDV4eOgA0jY1eD5ZuJGrkW99XgDcawxtpv+VtxC3SA6GU00xhfSMBl0mLyySs68R1nezmxMWDO5oIOpvdLjbktmdPGwwDWynGbKnisiUruGDFGg6pqeeCz5bz5xemc/1784jZFvnhGCMrG+lX3UB2KPn2uz9mMXhLopNm79omfMFtLjI0jayRyaOe3DDUYajdRUdZM/kEOmWNTdOUjSmzxTsEz1UZfjyd3J7XTYe1fXL5/Q+/w1+/dwxRs/10pKNwYSUd73QimNgkcjEOOg5ZdiJTbrUuG3B7CJmu9nboOrVpaShbUfV26gV8RHOx9a8VdKdmPys/Tu3MmtYrjYz81FIoI64wLYVtpv7I1M6q6rJUxp6yPHXau8tp/mBjUgdegNCCauLVXXdm3llWh+06reFmknCc6McbUOH2tpvEyaaePGpIpxmttZ9BcEpJp9tQm+pRXyU/UEwDHLaXbbbQ/npB6uR3v0iZ5Kexvf31EUwSn3HVyU1kRetdgHcXp7yXYuqXKNshTurdEQ2F5jfJcja3XnA4FLKedFI7RFOf2gdCTflqx9vfE95d1Mm01GMohDh47HLQHo1G+eCDDzjppJPw+RK3jc8880yqqqr47LP2bNL69etJS0tLKpnpyrp16wAYPHjwrjaJSCRCY2MjDQ0NrFu3jr/+9a+sXr2a4cOH06dPJxmb/ai+vp5otD24CgQCtLS0n9hjsRh1dckn/4qKii5fV1ZWJpUMfZu2MXF0hItzwxxa14DbSR11Ir05wMBVpeTXbK1X1oi5DCxgbWEBFWlpjFi7ieyWIL5IHDNm82nPIv7wnSNYl5sNQIZlo5kagYw0KgvziLlMLLcJhk5uUzMFtfUUl1UyYG1pUhC/VaPPR1zXcTSNmKETNnTcrXcAnNZbBf64hT/WXrMc1zQGV9VwYmkZhcEwfRpb+NG85UxcWkKPhhZGbqrmtHnLcUdjHFVez/iyhk572ddmJsoAGtJ9OJ1k77MOab9QqKioYOnvFnDTEx9xx0uf0L2uOWX+bTfSO0Ph7pN656ukMJsLbrqQq68+izUF3fB2Uvccc3QKm0NcPW0xM0b14ew7L2ZdYQ6gUEDQ7SHm1om5dGJunYhporV2WN16YZRGFBcW2taae02jOj2T8owsyjMy2ZiTQ6y1M7q3Z2pJRPbE9hp1Vyedhn3d/WiGnvRZjdVHiVeEU+Y1WoNrrZPBT9wFHkLx9kB72++H1ic1UWH3zMTdZ9tiHzCyPRhZnj3yHQzntwfOeocsckdO97S2TLuGTQ51eIhiYpFGkEyaAHD1zex0G5XxFkhPHeZUo/M6a1WcTaB/+/Fo248+3VLmtToG/hqY3bd+FjsbgaZ13/rk7/hY9SloDchTL7TSfn0sBfOvwK21963KoqY1C7/NFvVOyoW6J3/nOtqj590+Banb7lNw0JzbZRv7bhsHEpW4LE76J9rtctD+ySef0NLSwllnndU2bfz48eTk5CSVyAQCAdLSuq7h7Ghr59Idlb505rHHHmPChAmccsopXHzxxbz66quceOKJ3Hvvvbu8rr0tNzcXT4cOhunp6WRktP8wu91u8vLykpYpLi7u8nVRURFah6duftu2MekH3ehGnJjblRRyZBS4GTbIjatDljeQnU7aoEzcjkNuIEj3muShDzVgQH0zjq5TaNn0jNuMCIVx8jykBYL4wlE2FuWzpnshW3KysDUN07JIbwqgqcSteNXhHGMZOmt7FCZtw9J1XP0SAU44zYuja5hKcd6K9UnzHV9eldK2uKc9i2w6Dvl1iYDJ6yjsbQKE+QO7syU3sR0ny0PepckXxH0m9SJ3eHbb62x3Lps/TZRG9Kpv5rIZSzE7HLt+aQ6DMtqPsMuAu0/1kP+TYXgGd6yxVpR2y2J9US716X76VTfR5PLRsRDG1jQavYmL/oxwjCPXbiHqdvHm+CGJLKzRWnuvaYlyFU0jaHQW+IGPGN5thiA0emUQdrvayoEKvlPIsAeOxEhvz8B6inwMfvhoMk5LBO7pkSieeIfOnrrGobeOBJI/qzWTN0MkcQenfZcTpTEAhgNp/TqcxzQY9ofDyMxqv+u37ffDvOxItEO7d9gBHd9fv0vm6X1IP7FH0r4V//FIdI+xR76D2bedidYaRGqAy5UcSHvOG0ra+H74TuqN/7R+eImibxPYe4hgZLvJueWoTrdRPKAPxu+TRxfTjh+EVtTJuV7T0O+5jIzMTo7VlSfB0PYkkNJ1Gmh/nfF/h+P6x/dB19ES9TAdVqzQsMDjgj9duONjNXE0nDwSP9V0vJAxx/Yg7c6T0Uf0RPu/09qmO+gEKUwamtTBIH7SkdDx4kTXMB78fvux2Zvn3TsvStSyb9U7H352+kF1bpdt7JttHEhkyMeu7XJH1MmTJ5OTk0O3bt3YvHlz2/Rx48bx0Ucf0djYSHZ2Nunp6SmjvHRla4C/K8tsde655zJhwgQsy2LdunU899xzVFdXJ33QxTdX36Fp/Opfh7D0syacqI0Zt/Fnmgw7IQ+3z2DTtHJKPq5G5foYMqknjRsCfLCsjoKWAO5wasbUcrkYHLe33kynyeejrrKZokiUFT2KachIfFZrsqA6K4OhFdU0FebijkSJ+BMZbU8khtIg5PcS8mxTBqAUF9w3ivpFdQTrY/Q8NIOmJfUc6cC1Y9zMrNYZkK8TL3NTt37buvnkU9iYYzPJX1JL1rIKPDELR0uU7mYdksmVP+nO0AEQN3QuGKzRI2MkVWcXU7Oglpxh2XQ/IbmAXXdpaIaGshRK1xhUVcdf/vcRm84bwaGnF3PJ4S404NXlNrUhxXeHGAzM0wGDoYsuoOGV9TS9VULw7XWc/eUaejY2s7Ioke2LGybl/mz8VhRlaATcHhy9PcCJtT411h+P4SNKGBfbUmjEdQP3NhnxzPGF9L31aFx906maugVvkY/uk3rTvKqR6hlVpPfPoPj0HuimznHLv0vla6XoXoPiC/viyvHQb8rZNL+zgcjKBgaML6axKk6oIkzxKd3JGpLa4VP3GYksrAWq9UlcuqEx7PFxWHVR8s/qiX9QJhVvbya0OUi3U7qTNaLrkj8t3YN3/o3Yr32JqmjGOOdQ9CGJi72BU8+hcfIGouuayDylF/4xqRnn3aX3zMa//DdYLy9GheP4vzcKq7SJ2KyNmCMK8ZyZGC5R0zSKp5xH4EYvPPhx8kpMnZ6LL8fVd/v7aPzqVLTjB6Gmr0YbUoh29ki0ljC8MidRb+3zJDpOnjkGRmzn7mhWGiz4G7w6F2qaYdIRZG6M4VlYiefIYnwn903MN6ov2pSFkOmHCBC3wVCJzuHnHQn9duL46Tp8cDveyZ9jzi4hZvnQjx6A5/zhaJ7Wn81/XQXnHgWfr4WhvbEveoeG2AA8JO5QRckk/cqTcd06EeefH0FWGvod30Xrm7/j7e8J3xkOKx+E1+dBhg8uOjZxTIQQBy1N7cI9kvLycr773e92eVvlxhtv5JJLLuHqq69m0aJFvPXWWztVIrN1zPUzzjiDP/7xjzvVnq3LXH/99Vx22WVt05csWcJPfvITJkyYwN13371T6xLfHrGQxbOXLaC5MoIRt0hrCLTlgGvTfEweM5x4h0A7PRqjV1MzQ6vr2NgtL/HAH7+vLeg8rKUWoy6cyAq0Zoc7WpOXQ1Vme2ZkQI7Dv//Wa4ftXP3BFj68rb3+VTc0fHUBjNYaa1+Bl0kfnsqWqeUsuKG9w6TpN5nwwSlkDtr1/hyzb1vE6pdK215n9knj3PcmYHo6HxVkW1ZtmLUj/odV1V4KUuHNJNqhzlzPMIhH2rOgW3LSueGqiZgo/nv/6/SvbsBGo8adjtLaA3vLrdNjqJe0z9tr6C0P9Ft0GWnDOikF2EvssMWckW8TLmnv1Nzrp4MZ9vC4fdaG/UUFowRH/B1V2j7qkOvn4/E+eN5+bNWBIfCzt4k8PL/ttd4vh5xlv0Dz74ORYoT4hijT/pQyrae6fT+05MC0S5n2d955B6UUt912W6dlLI888ghvv/02l1xyCSeddBKLFi1i8uTJ/OxnP9vhukeNGkVeXh4zZsxoy9bvrlGjRnHGGWfw7rvvcvHFFzNq1KjdXpf45nH7Tb7/6OEserWMxvIw+cVu4ptbeHmdyayiIgzAbL0u7dXYzNGbytE0sEyTvvWNAEQNg1XFhcRNg1NuHEjoq3pK3yhli5V6d2dkyWbWFuZTmZNJVYaX06/qnjJPZw45rTveTBer3t2Cy29y6Pd60bysgc3Tt5DW3c/QKwbhyXbT76J+eHI9bHxjI64MF4OuHLhbATvAMXceRt6wbLbMqSarXzrDrxi40wE7gJnvY8D8C6l94EviZQEappaT3RIh6HKIGzou26HP5YPQxhRR/WklVQXpLBgzmCvyTa4dbGPfGWgto9HIjoUJmW5UYRqugZl0v6Av/X88mNDUDTT9byVrqjaw5UQ3hwzK3q193V2Gz+So2aez8Z8rCa5pJv+U7vT8ycB92ob9RUvz4J/7C+IPzMBZV4tx2hBcPz5qfzfrgJD24FmYo4qITV2HMSgP3w3HSMAuxC47MMt2DhQ7nWl3HIezzz6bjIwMXnrppU7nefzxx3n88cd59tlnGTBgAD/4wQ8oLy/nnnvu4YQTTkiZf+XKlSxbtqxt2MfJkyfzpz/9iSOPPJK///3vKTXx0WiUJ598kh/+8Iekp6dvN9MOiQc7fe9732PMmDE8/PDDO7OL4lssGlcc96sawhZoSuFSiVPH2SvXkRaPY5kGSk/uAlKZmUGt38cdl/spGJzJ66dOo644LylLr9s2hZuq0YCnjx9JyZF9KfmJgaeTEUa+iTbdNJeq+5a2T9A1hs0/l7SxnZcolP58NtX/WtY+wdAY8cX5+EcllxTE43GeeeYZAK644gpcrtRyGiGEEAeXMu2ulGk91W37oSUHpp3OtM+bN4+qqiomTZq03XlOOukkHn/8cSZPnszvfvc7HnjgAa6//npuvvlmxo0bx1FHHUVWVhYNDQ188cUXfPbZZ1x++eVty0+aNImqqiqeeOIJzj33XCZOnEj//v1xHIfS0lI++ugj6uvr+dGPfrTD9vbq1YtTTz2V999/n8WLF3PYYYft7K6Kb6FNq1vYOqqd0jTiKFyOIq21Y2JnPdgzQ2H6bNxCcPMAvL5ENjq7uoGWnAxiXg9mLI4vECackehwef7yDYy/vgiPueudrQ9WPe8+Cs1t0PDKesx8L8W/O2y7ATtA738cje4zaHitBLObjx63j0kJ2IUQQnwzyWgxXdvpoH3y5MkA231QEsDAgQPp3bs3H374ITfeeCO9evXihRde4PXXX+fjjz/m6aefJhQKkZWVxdChQ7njjjs47bTTktZx9dVXM378eF5++WVmzJjB66+/jqZp9OzZk1NOOYXvfe97Oz0qzZVXXsnUqVN59NFHeeyxx3Z2V8W30Np/ryQ3Ukh9WqKjltI0LAO6jciielkTunJwSC4Tya9rxBuJ0nN8NzJ7+nGlmRC0yK5NjOhiGTqWr71cxhW1qHp6NfxzzL7bsf1Mdxv0uvsoet29cyUUuseg99+Opvffjt7LLRNCCHGgkdFiurZLHVGF+KZ65pgP2GKbTBs6gLr0NHyxOMeF6/ntXQOY+pdVbFnejPKY2GhoSuEORsjRYhz5i6EMv6QfAJs/rWTO7YsJVoRJ7+EnHINgOPnr5c1x8eNPJuyPXfxGkfIYIYT45tmk/TllWm91635oyYFpl4d8FOKbKH9YFtF5tVy8cBkhl4nHsqktyGXO9EZ+8NjhhBpjuP0mVsxBNzTssIU73YXhbq9z73VCERd+ehrRxijeXA/LX9vMjD8nP+myoJMhBIUQQggh5TE7sssPVxLim+iYXw9H8yeuYf1xi4jPS0tmOssXJJ485892Y7p1vOkmbp+BL9eTFLBvpRsavjwvmqYx5Jwe9Dq6vR7bX+Dh2JuG7JsdEkIIIQ4y8kTUrkmmXQggb3AmPf9vFPNeLMM2dKLeRC16dv7ul12YHoNzHjmC6hVNRJvjdB+Ti+GS62QhhBBC7DoJ2oVoNf6cAr6Y00yoLjFijOnSOPWir//0yW7DpCRGCCGE2DHJrHdFgnYhWmXlurj5/kEsntVINOww6pgs8ork4ShCCCHEviAjo3RNgnYhOvCnGxx7et7+boYQQgghRBIJ2oUQQgghxH4nHU+7JkG7EEIIIYTY7yRo75oMZSGEEEIIIcQBTjLtQgghhBBiv5NMe9ckaBdCCCGEEPudjB7TNSmPEUIIIYQQ4gAnmXYhhBBCCHEAkPKYrkjQLoQQQggh9jupae+alMcIIYQQQghxgJNMuxBCCCGE2O8k0941CdqFEEIIIcR+J6PHdE2CdiG+gWrWBVj86mbiIZshE4sYMD6fWNDiy5c2Ub2yicLhWQw9qzur3tpMzfImuo3IZuQlfXH55ZQghBBCHIjkF1qIb5jakgAvXrUQK+oAsHp6NRNvHcrKyZvZsrgRgJIZNXz59DrizXEANnxSxebPavjuU0fvr2YLIYT4lpPymK5J0C7EXjD340YWzmnGn2Zw0lm59B/sw4o7zHijmnVftpDfw8MJ3yskr8izx7f91eQtbQH7VnMeLyG8JUjc7SLmcaPbNmZjS9I8WxbWU7uqifwhWXu8TUIIIYT4eiRoF2IPmz6lnjf+W932+qtFAX79lz7MerWKL2c0ALBheZBVC5u5+eGhePzGHt2+FXNSpjXXxrC9HuLexEWC7qTOs71lhRBCiH1BMu1dkyEfhdjDZk1rSHptxRX33rmZL2cmT2+pt3j132U49p7tejPs9KKUabZpYLld7W0yTWwj+WLBm+dh6r1refX6L9m4oH6PtkkIIYTYEdXJP9FOgnYh9oFg0EF1cvZZMruRaa9Up77xNfQYmY2/Xya2ruNoGnGXiWWayWc/TaMlO4Oo10PI5WJzRhoNYY3adUE2LWzgjZuWUr2mZbvbEEIIIcS+JUG7EHvY+AnZAJiWTVokQkY4gi8WJ7pNZtvWNIIuF59Pb+hkLbtHKcXnz28k1hwn5vUQ9XmxTJOcmkZ6lWyhx4YK0ppDADiGQSAznadHDcFlGihdxzIM4i6TmKYz/4XNe6xdQgghxI4otJR/op3UtAuxh004O494wGLWC1vapvnjcUIuFyGXieEoLF0n4PVg6zohzdpj2178WjmzHikBwGXo2IZBbnUjGa2Bumk7dNtSS6WWj6Pr+JqDlA/uh6Xr2K2BO4DSNFbNaeCkhhhpOe491j4hhBBieyRI75oE7ULspoUzG/ng5Rq2NDmE/V4sNI4c4+fHl+VCJDUQ98bj1ORmEneST0pGrnePtWnVtKr29doOhu3gD4ST5tGA/OoGzLiDrhQ//Xw5i/oXcUJ1ch27bSleu2AWJ/9uGD1PKt5jbdwZdshi3Q3zqH6xBDPHQ+/fjaLHNUP2aRuEEEKIA4mUxwixGzavD/O/B8vZUm1RZ3gIxSAWU8z+LMiTz9URbkkN2nVDwwzFUqb37Nd50B6L2LvcLk9663W4Umwtom/LntNe1m4bOsH0xEgyh1bVc90QB0dPzXCsChl8fO1ntGwK7HJbdpdtOaz99QIqnliDHbCIbA6y5tq51H9Uvs/aIIQQYt+Tjqhdk0y7ELvh4/cbaHK7iZomaO3Bri8Wp/TDSsocBbqO4ThtN/uMaJweoQjBsJuqrEwcXUdTivCyWmo3Z5DfywfA+iUtTH5kM7XlUYr6ejn3573pNThtp9o15uKebJ5XC3Zi6EalaQQyfXhDUUJpXpShY8YslHLA0JkxejCbivLwbrbRC3TGVrVn26u8HuZ1LyTD6+Lxa7/ihJ/2Y8w5qSPT7CnKUUx7fCOL3q3GCkO3YcUMWFHZdrt0xU8+48iFZ+HO33N3JoQQQhw4pDyma5JpF98agWaLYGDXs9fbisUc5i6O4GgaWofxzg3HoSAUwrSdtix33DBocZm4YzHcto0GpMdidG9oJDMUokdDI83rArz653UAREM2z/+5hNryKACVpRGev6sE2+o832DHHVoqwygn8X6gIoxmJy4UNEBXiqjXRTDDhzISX3fLbWK7TGxNY3NhLmganmichekZvNi3JwvysvmgeyGPDu6PB0DTiEYUU/9ZQvnK1BFloi1xQnWJ9gYrQljhrmv043UR4vWJ+W1HsanRwbIVX0ypYv4blcSjDqBwDA2F3ronGpGNQVZfN7/DzoNZT9u+VwYUzdGu8zL1IUVdKDGPE7aIle3GHYRgBMrrAAhXh4m3xJPfbwhAbXPqcmV1EIru+vYAGoNQ07T99yMx2FybNMlujGDXhHZve0Ic4CqDipbY3s3DBmKKioDkesWBY69l2hcuXMg111zD9ddfz2WXXdblPNtjGAbz58/nlVde4W9/+xs33ngjl1xyyXbn/8Mf/sC7777LU089xahRo7jjjjuYMmUKH330EdnZ2V93l8RBKhpx+M+/KliyMICmwRHjM7n0p0WY5u5d0S//KowdsnADLsBUirBp4o1b6I7CbA3OFYkyFMNyMLYZ79FrWdhxi3hrpr66NEzFuiBvPLCRaDj5AUctDRYvTW3kB2fmJE1fO62SGX9fSbg+RmYPHxPuOJQNs5MDNwDdUUl3AxITddZ1y2srnclvDnBqXQNPDRvE0txsADJicY6qTa5zXzevgR5DM4BEsDz7bytY/tomVMQiQ3ewW+K40k1G/WIYw68anLSsHbJY9cOZ1L5eCrqGfVY/fnzUWEqDOsUZGr+uSGyrR2Udo1aVEo2bhEjOqle/nRjRpv7NjQy+BVzNGp88+w53XX86s1vceAz42WEa/zhBR+uwz1FL8eO34rz4lY0C7t60gjPeWIjTHMN7aB59XjoN77DclGOX4p434K7XIBihOacbs4wjCKdlMvBHAxn12+FoP30Mnp8BjoLvHgnP/QLK6+Gi+2FJKWT44M4L4Zdn73hbAJYN1zwKz36auHty1hh4/gbI9LfP8/iH8Jv/JgL7IT1Qz99AzSNraX52OdgO/rMGUPT8meiZe/7pu0Lsa5VBxcXv2MwoA48B1x+u8dfj9+wD6gD+MMfm7wsUYQvGFcMrZxv0ypQs8N4mmfauHRDlMRMnTuTYY49Nma63BhSnn346//znP3nnnXe2G7QHg0GmT59O3759GTVq1F5trziwNTfbNDXZ9OjhQtc13nujjsULAli6huEo5s9spmdvDxPOziUec6guj5JX5MbrSz7xN9bGiUZsCnsmB45fzG1uu0WlAW7HQVkWIw/zUzOjpe2Uo5HoDBrvpFbc0bTE+0oRc5l4TcXH/y2nan0QzOSvpQPc8mGUQwdEGNzXjders2lFC+/euRyjtcNrc3mY93+9mAGnpnYY7SxPpDQoK2gPUiNuFyOqa7l58XIWF+TitWyGNjQRTvORYdlt+5tV2B74LfzvBpa8vBHdUXgjcezWkpx4wGLhX5ZSMDYPleUlI8+NP8vFpr8soea1UkLpHlxRC/fkEo5uzqD0mKFUtChm1JkcHY0xdtl6dKWIkfpDHEGj5tMK1v1wJpZyoRk2940eweyWxAg3URvuW6g4slhx0ZD2437/XIv/LU3cZTmkso7T/jOXrZdGka/q2HT5NOz3LyDTo9Fjez/Ms1bAb59ve5nZUM1Y9wJmmCey5rE15JZvoMezM2g2M0kngOvN+fCnV2HmqkTADtASghv/A6P6oI4YjCqpRTukENwmLSsb8RT68Gwt/1EKfv8iPDW9vQ3vLIQ/vIT6w8WojfVouoN2zWNtd3ZYVU7TWU/RXNl+gRd6Zz11f5hDwf0ndb5fQuymcFyxugEGZkO6e+8EW5ubFcE4DMlLrP/6jx1mlCXei9rwtwWKo4odzhu8c4UDSilW1EGeD4rSOm/z1A0Of/ys/cw5rwKu/chhynl7/uKgK81RxYYmGJILnt1MMh1s5L5G1w6IoH3IkCGcccYZ230/IyODE088kQ8++IBVq1YxZEjqKBLTpk0jEolwzjnn7M2migPcKy/X8/77jdg2FBSYXH99IZ9/EaTO5010tFSK9FiclctCFBebvPRQOcEWG49XZ9KVRYybkIttK158qIzFs5pQCnr293Llb/uQnZd4omhpSSRlux7HQQ/HU3IEicAc4rqOq7WURkEiw751HqXAY7BmfmOipMW2cTqM6b46PY0jmyLcc3cFfjf0dVvUV0ShWyH+UJiC2np0INwQZ90nVeimhtNaThPXdT7r15NjNmzGbdlt2w/7vPRuamZNQR5K06jIzqRvTR3dQ2G6byzn7X69+MfYQ3F0ndxIlB+s30j3WJThJ+UTqI/x+q0rqFgVgKwMzGiMtObkEWoA3rz+S2rdfgyXxjHf74HzcRULjh1IxO9GcxQ9N9YxprSSF48ZCsDMgnwu/2ghemsAavs0osrAE2lvd01RGnO++zF1PXOIuw00pfj8kO4p2/5oo+KiDqeJj0ra714cuSG5Q2tlZhrXH3Esqx+JoQGXjdR5+hwXxrYXW+8tStlOYawaTTkoTafkkwYWFp5DTPdgOnEOa/6SAVO/hCWbWudu71ZlnfIP4mYuxGxacvJYmj2ccE0MzaUz4BdDGXpZEZxzN6xIHSvffmEh9mOLIRyHTA+mMtFpL9EJV6b+7IWnbkiZJsTX8dZahyunOjREIMMND52k88MRe67iNm4rLn/f4eVVCgWMLYS3zzX4aGPq5/vJr3YuaF/boDjnTZtV9WBocM0ojYdOTr4r5yjFr2c4Kct2tt296ZEvHW7+1CFkQYEP/nuGzsR+UtH8bXfQfAImTZoEwOTJkzt9/+2338YwDM4888x92SxxAFm5MsyUKYmAHaCmxuLJp2rYFNDbR0bRNAJuF94MgxceTATskCihee2xLTTXx/n84wYWzWxqS16WlUR459nKtu0UFqeOW647DstWREg91SfqyiNuDy1eLxHTJOx2Y3cIyhUa8RYLlMJwHNyWhTsWw2z9b5oOntbGmM3RRMDeKuT30ZKR3roeCFVHqcrwsyo3hy+LC5k8/BA25mbz8aB+WKaBbeiJiwa3m7xQhNHlVfSpb6R/XSOumI0ei7M6K4OP+/TAab3TVe/18FafHvQelobLa/Dp46WJgL31eFoeN04nSaBwMHFs7bhi1nNlLMnKJuJPHDula2zul09Ft6y2+UduqsYTbK+H99g264cWUN4ni+ru6ZQMzSOQ5aE+30fcnTh+StPIb0m9YHCjsOz2H9lD8tsbuDEvK2neeyeOY3VxftsxfG6pw3NLO+n7sCZ19JqY5kJpiWNaE/IT0xN3IizdxcKsMYT794aeeXQM2BU6cScTYoltrIgUE65JjCqk4g7r7l1O3eXPdxqwKwzsaicRsAM0R7HITMpOuYzUEYpcKvUYCbG7gjHFD99PBOwALTG4eppDdXDPBbZPfqV4qTVgB1hYBb+Z6dAvK3XeTzdBKL7jbf/sI4dVrRV/toJ/f6l4a13ycq+tVixNrTLEuw+T7BubFD+fngjYAWrCcPn7DjF731447A/ycKWuHRBBeyQSobGxMeVfINDeSWzs2LH06NGDqVOnEosl/yht3LiRpUuXMn78ePLy8vZ188UBYs3q1Ax46YYYkW07K2kaaVkm4W06pTo2lK4Js2Flaue9DSuDbf9/+GF+tA5hkqYUugJ0naDXnRRAqdbtGY6DZZqEPR4svf1r52iJDqymbbeVzGwtm3HZNoZSZEXbs6guOzWYjHgSgbDmOHiDEfqtreTI1Rs5Ztk6+lUlOkzWZvjRbZu0xgBpzSH01qy/z7Lo0RygWzCEMg1C6WlsTEsdqWZzehrH/qQfAGVfJXey1Byn7WKgfb80HKP9ZKuAZjv1dLOqX0Hb/5+4voyYadLk9lKTk8HMI4YSzvRT2SeXyt7ZxL0m2TUhYp7kG4Tf/WIV3g7HyB+K8vCsKIPuD7O6JrGfvxlvkutLHNzZg3oxa2CvtvkX9y5MadesTZ38ODYGUyZFWoN0nxXF3uZ0qjSd2tNOhPt+CB0u0hzciYYANhotRnrKeuuWdtKRFXBwdTLVaP0H6DrZPWOYtH8XdOLkZnQShQixm76qheZtrg1jNnzeyV2e3TW7PHVds8oU5w9KDeLCNizbiY/4rO2sc0fbBeikynGv+axCsW18Xh2CNfWdz//NonXyT2x1QATtjz32GBMmTEj5d9ttt7XNo2kaZ599Ns3NzcyYMSNp+XfeeQfgoCiNqa+vJxptz5QGAgFaWtpH5IjFYtTV1SUtU1FR0eXryspKVIeOjt/WbfTslZoBz8/XSE9PTZGMHpuOa5saTE2DrHyb3NQYjqI+iTpjpRTvvFwJtoNh2bhsG7fttJV0hN1ugi5Xe1619bar0toz/bZhgJMYYUa3HXyt+xh1uzvN1Me97UFqx4B/K1fcQgFm3MIVtdraYjqK41asY0hZJaNCDfiaQ2ith9cV22bEE6WwfS78LUF6NqeOEHNInkZGd4toNEpBf3/Se65oHMfQibkN4qZOzKUTd+lY7vYAUwMy8lP/Pt/7YhkXrS/lusUrKGq9GK9Oy2BZv55kN4TwB6OgacQ9bgrLWigqb8GMJV+4DKhp5PmH3uD3r8zgb89OI6M2cbFf2qi46YMYlZWV9MrSeO+yxPYdXefn3z+VKYcOoiojnZ4NqSPIDEhPvnCrqKiAUf1S5ovjZ3jDRg6vW98xLm/b6ayT+8MFxxD84m5Uhrd1cvvdBAOFz0nNgmf27XxYy47Ltsnxwwe/I/LPKwh8cQ/macPpzRIKWUM31tKHxXiO7tW+H9vuVwcHw/dctrH/t5GnGvBs83nXNRjRekdrT2xjgD/1ezGyQOPCIannQJ8JmbHqHW5jeG7qGXZkQfvvQEVFBYcWdB4kHlGs7bO/x4hOco8ZbuibtXf+5uLgoam99JfbldFjzj33XCZMmJDyfk5ODoMHt49AUVlZyTnnnMO4ceN48MEHAbBtm7POOgvHcXj33XcxO9QKy+gx3y6Oo/jXQ1UsXJgIuNxujetvKKQ+oHj46TpaS7o57aR0rro0lznv1/HmUxVsHbVxwvkFnPGDQqJhm0fuKGXT2sSPRnqWwTV39KOwl4e/PVDNoq8SWUxNKXLCETytQw5agGUm6qyzQpGkGvaoaaI6pGrcloXuOBi2ja4St4AjXi+mZeGNxdpyC71HZ1F0Snee+189SoGpHLo7MWKhxM4U9/EwQAtS/mkF7kgcd2SbYJxE1lttkyayXCb1hbk4ZvIvrxmNkVHXyBNjR7CkKJEF9+Lwzk/SmDAo8d2q3RjipZuWEahLBNn5RgynbJvMsA4NfQqxWpsz/KR8Dj21gFd+uwJLS/zo9qiq4+ilq6k/uh/z0vJxIjY9NtSS1RwiKxBJ3L0AynvkUDKwiL7ra8gvayaY7qKiZ3rbPuU2hehf3gAaPD5+FE98p70jerc0qPpt+52DGz+Icf9niWN37ysf07u+hdK8TP50zrE0+xNZ86N7aky71E3ath3rqhrhhNthVaJMxnZ5aYj3wsGN3jeb+p+fwuL7V6FaU2RDfz6UUb8d2b78K3Pgsn9CzCJOJhaJUXjqXDksTR+C3Xp7v3hSL8b8rBDtzD8nho/sqHc+1vHjcP77eeK1y8B46jKMy8a1z1NWCyf8Hta3lnQN7g6f/hGKd2J0HCF20oOLHH75iYOjEhflvz9a445j91wNSXNUMeFVmwWtH+OiNJh+gcGwfI3bZ9v8eV7ivKlr8M8Tda47fMc5yLnlijPesGlqjbNP6aMx5Twdd4e7ghFLcdprdltnV4AcD3xykcGobvsu63vzpzb3LkycEwwNHj1F5ycjD4g86171hfZIyrQx6tr90JID0wHREbV3794cddRRO5yvqKiIcePGMW/ePKqrq+nWrRufffYZNTU1XH755UkBu/j20XWNX1xfRElJhLo6i2HDfKSlJX5ERg71snJtlJ7FJr17JjKux56ex7CxGWxcG6Z7Hy/deiSCNo/P4Pp7+rN+eZBIyGHwqHTcHp3PFgTbAnZIZM+bPW6KAiF028atacRtGw2F6p6GP9dk3YYY2ZFoUkcnlEp0PtU0HF1Hdxw2ZqaTH42DaRI0DAzbxtF1zr5pALmFbg4/zM/6khh9erspyDNYvSSAy6UzcEQauqGx7LVcZt25BEXyzUTL0FNuLm7NyheUV1PTvQDH1SGT73FTU5DLGRu3cHJJGS1uF2MzbSYMGt82T34fPz/93xhKv2jE5TWIl7Xwya+/SNpG8eF5XPrvsWxc0kx2sYfiQYkSkMtGWqx+vgR/JEZucyIgHTg2i8P+eBSbv2wiI8dk03Fv4XRIJfQob6CqMJsBo32oskoyA1C8tpEWvxcrrjP4T4eT1SedyVoWT3zpS2rH2B7JQcR9p7n5yeEOy2sU3WryqHm9hb51zTz83w9Z2quAYef15PIrBiT/vbYqzIavHoCPvwLLRj9hOBmzy8BycE/oT77bpMdFA6lbVEf20CwyB2YmL3/hsXD8cJi5AtfgYgyPD/VVBd3H9aVbmp/aWVX4e6aRPaY1zbbxMfhoCeRlQDQOMQsmjML0uHB+PRFWVqKNH4hWvE2Rb898WPkgfLQ0EdGcPBLMfViQK74VfnG4zqQBGp9XKkYVaAzO3bMBbaZHY94PDGZsVrTEEgG2z5XYxp/GG1w2TLGkRnFkkUafrJ3b9jE9NDZdbTB9k6LQr3FMj9TlvKbGpxebzNrsMKNMMTBb45yBGn7Xvi3T+McJBleOUCyvUxzTXaNHxrejTETy/1076KLcc845h7lz5zJlyhSuvPLKg6o0Ruwb/ft76d8/eVpOtsExR/hT5s0pcJNTkFq2oWkaA0ck1xpvLEvNYluGge7YxFwurK31EUqRZjssK1PEfF6UppETibaN3b71IUtbl8fU8QzMZsPmCP2aWzA0jZhpEumRQW5hom0F+S4K8tvLTYaPTQ4IR3yvN43rmlnx7HrMqIUGhNO8xN0u0pqTa7E1IOp1YVhO2wOXOqrNzaIkLweXZTFuxXqcWOotZZfHYNAxieDSGZnJxk8qKXk/kYFOK/RyzG0j8We5GPqd5Pu8xbcegT27jMjnieJMz+gC8m4bh5nt5pATCgivbaJ024cVAYeO8HHo7SNYt6QSa3MAbEhvibLlMA/FPxuCx+fhEkvxRiTKO6sSmfS+2Rr3np76tx3WTWdYNwjfPpJZy+ppXt2Mx7I5vdDm2Gv7dB6wb2UacOrotuPoOXVg0ttpPfyk9Uj9nLUpzIYLjgFaaxOHJJ4w6wa6T+qdPG+GD84dR2f0ET1gRI/tb8dlwumHb/99IfaAPlk7HzDvDl3TOLF35+sfnLt7FwqZHo1zO6mL39ZxvXSO67XD2faqYfkaw/K/HcG62DkHXdB+/PHHk5WVxZQpUzjvvPOYOXMmo0aNom/fvvu7aeIbbujg1IfT9O/tIt1ws6WlQ/CradQ32fgNMB2HoNtF0O3CbTtkhSP4bQfVeh52dB3b7+LYI9N4phaaPG7S4hZh0+Ds4zJ2qX3jbxnByEv7U7eqkWhznCn3lqA7Tkr23dE1Qhl+jNa6fMulJT18yRW36FVbT1VWJqv6dOf8oq6fqqmbOhPuP4LGnw8hXBelcHQuuqvz27hGro8+8y8hsrAS5Si8RxQlBcnefhm4e6QRK+9woaHBkbcfimdgFsM2/JDArC20rGtm6pY5xItsdDOxLbep8falXpZVOTSEFUf30jGN7f/g+Yr9nDLjdOoX1qF7DXIOzdnuvEIIIfY+GS2mawddgZTL5eKMM85g06ZN3HPPPcTj8bbhIIXYm0YN93H2xMy2DoeFBSY/uyqf4cdtUyusFDrgs22yYjEKQmF0pehpxMhr/X/DUaDA1nViUcXJo9wcfagHW9dp9rgZeoiPH529a0E7QGZPP/0mdCdvVF6i/MYwCGf42m45OppGS3YGlsdNNM2HMnSM1k6xALpl0bumjqEV1Ry7pgTN1Dj+V0N3atvZ/TMoPiJ/uwF7R96xRfiOLE7JamumzqDnTsDVLVHmovsM+t5/NL6BiRIQzdDJOKEnBT8cRLyo8xupIwp1jutrdBmwt21P08g7Il8CdiGEOACoTv6Jdns9075gwYKk3tVbZWdnt2XHV61axXvvvdfp8ieccAJ+f/Lt5kmTJvHiiy/y0Ucf4ff7OeWUU/Z4u4XozOUX53DO6Zk0Ndv0an3iqnlSNtPea2ybZ9uRZXWgIBLGHUgeDcForW3P7uame28Pd1/npaLWwrKhV+HX+2pGwu0Z9pjfy6acTErT/MRMg6MamtC3CZYPm5hP5YYgLYvax01zOQ6j9SBZvboo99gLsk7qwZjNlxBe0YCnbwZmduodDiGEEOLbZq8H7XPnzmXu3Lkp0/v06cNvf/tbAKZOncrUqVM7Xf7NN99MCdoHDhzI8OHDWb58ORMmTMDn83W6rBB7Q3aWQXZWe8e+7r08XHxlNya/VEs45OD16sQiyXXgjkOnKYPMLIPLb+qF3joSSnH+nvlKttTEcIei2G6Tpfk5/HdQX5zWQH1wKEzeNkM+xiMOOabDtoM9Frg6ecjQPqC7DdJG5++XbQshhNg/HCmP6dJeC9rHjh3LwoULdzjfzszTmWeffXaH89xxxx3ccccdu7V+IXbF8RNzOPrELALNNl8tbOHVp6va3tMdB1vTiBk6Lrv9lKSAUy8ooM/gPZ/JLujhJq05iAI+GTaoLWAHWJmVwfia5Kd0ZOUYZA/OZ9PnydMHHieBsxBCiH1Datq7dtDVtAtxoHK7dXLzXRx3ag4nnpGL6dLQHQeX4+C3bSyXScTd/uAlpWnEQp09Tunri1QkOo9qrcNSdjStqBubPe5EHbtSuCNRlj29DhWxGH1hLwyPjuHSGDGpB2Mv67tX2ieEEEKIXXPQjR4jxIFO1zXO/1EhZ16cz+8uXZn0ntI14oaO6Sg0XWPouKztrOXrqVxY1zYizGGVNXzSr2fbe3FDZ1Z+LlcvXwu0jyyz6NkN/OTjkznu54MSD3JyyzW9EEKIfeeb0vG0vLycmTNnUl1dzfnnn0/Pnj2xbZumpiaysrIwUh6hvXPkV1mIvcQ0NPROTkFK03A0jcNPy6db773THyPWYazz81eWcNzGLXgsi8xojO6BIL0CITSSh4KMBSyUozBcugTsQggh9jnVOpRDx38HE6UUN954I/369eMHP/gBN954I2vWrAEgEAjQt29fHnrood1ev/wyC7GXmC6dkdtm0hXYhoGW4eaUHxTttW0POKtnW0TusR0uX7qG+6fP4655X3LLF8s5q6IC3Uw+GQ6aWIymH1wnSCGEEOJA8fe//51//vOf3HzzzUybNg2l2hN3WVlZnHfeebz++uu7vX4pjxFiL7rw/3rgSzNYvqAZx1FgKwp7eznjB0VkZLt2vILdVDQ2nxPvO4Ilj60h1hyj94RignGdzfPryOzp4+ifDcIK23z+2DoC1RH6Hd+NY64/ZK+1RwghhNiRgy2zvq0nnniCyy+/nL/85S/U1dWlvD9y5Ejef//93V6/BO1C7EVev8EF1/bggmu7eOT8XjLgrJ6JjHsX+hxbsI9aI4QQQnTtYK9p37x5M8ccc8x2309LS6O5uXm31y/lMUIIIYQQQnxN3bp1Y/Pmzdt9/4svvqB37967vX4J2oUQQgghxH53sHdEPe+883j00UcpKSlpm6a1juT24Ycf8p///IcLLrhgt9cvQbsQQgghhNjvDvag/c4776S4uJjRo0dz+eWXo2kaf/3rXxk/fjynn346I0eO5He/+91ur1+CdiGEEEIIIb6mrKws5s2bx69//WvKy8vxer3MmDGDxsZG/vCHPzBr1iz8/t1/Crp0RBVCCCGEEPvdwd4RFcDn83Hbbbdx22237fF1S9AuhBBCCCH2u4OtHGZfk6BdCCGEEEKIr+nKK6/c4TyapvHUU0/t1volaBdCCCGEEPvdwV4e8/HHH7eNFrOVbdtUVFRg2zYFBQWkpaXt9volaBdCCCGEEPvdwV4eU1pa2un0eDzOY489xgMPPMC0adN2e/0yeowQQgghhBB7icvl4rrrruPUU0/luuuu2+31SNAuhBBCCCH2u4N9nPYdGTVqFDNnztzt5aU8RgghhBBC7HfO/m7AXjZt2jQZp12Ig0X5rCrWvbkR028y9NIB5A7J2t9NEkIIIcQe8Mc//rHT6Y2NjcycOZNFixZxyy237Pb6JWgXYh8peWczn97wedvrdW9u4pw3TiTnEAnchRBCCKUf3OUwd9xxR6fTc3JyGDBgAI8++ihXXXXVbq9fgnYh9pFlT69Nem1HbFa+UMIxdx62n1okhBBCHDjUwR2z4zh7t8BHgnYh9qJ4xGbp+1XUbQwTrI0S87iI+r3otoMvEGbT4kZ6LGmiz6hEtj1cG2HR7xfRvLqZbsd2Y/TvRmF4jf28F0IIIYTY3yRoF2IvUY7ipZuXUb68BQB33E2sMKPt/XC6j/o6xfqbVnDmTQMYdkw2U455FztgAdC0somKjys4a/aZ+6X9QgghxL50sJXHbNq0abeW6927924tJ0G7EHtJ6aLGtoAdIOZ1J73vmAZO65PT5vyvDGNNDVbAShrgKlASoGZ+DQVHFeyLJgshhBD7jTrIBiLv27dvyhNQd4Zt27u1PQnahdjDoi1x1n5QQemS5vaJqvOHM+uWhScSwwmFqDbc2IaGo2nEfG5AwxeM0LS+mcqlDeiGRv+ze+Er8Cato3F+DXVTt+Drn07OiUVseWczTsyhx3l9cGxF2dubMP0mvc/rgzvHsxf3fPsc22HL1C00rWwkb2w+RccX7Zd2AARbLBbNbiIadhh9TBZut8biWY0o4LDjssnKdVHT4vDaF1HiNpx7mJteuVKiJIQQItnTTz+9W0H77tKU2k40IYTYZZHGGK9eOpfmsjBKg0BmOmiQ1hQg7nET83UImpUio7YJdyyO5ihM22nLsluGTnN+JpqCnHAQp7VkxpPt5szXTiCrf6LMZtNDK1n1i8SINI4OsVw3jt36lfYZ2LqGE090jPEW+Zjw/in4ind/jNjdNffquZS9s7nt9eCfDmb0HbvfATcej/PMM88AcMUVV+ByuXZquab6OPf9poSm+sTxNA1wGxCLtB4jv853f9WXi16JUhdMHMc0N7xzXRaH95EchxBC7E3vpj2fMu3M4KX7oSUHpoPsRoQQB7blb2ymuSwMgOYoMmsb8YQiuOIWvkAIdziK5jjolo0nGMYdiwMkBexbX3vCMRxDJ6C1B6TRxhjLnlgDgBN3WPeHL9veszx6e8AOWHGnLWAHiFSGWbfNCDb7QsPS+qSAHWDtk2sJV4f3eVtmv1/fFrBD4hhuDdgBIiGHF/9T1RawAwRj8I9poX3aTiGE+DZydC3ln2gnqSMhdqC5IsymubWkF3rpdVQeZXOqCVZF6HVcN1o2BWncEKD4yHxyBmQQqIy0L6gUulJ4gxGUBhrgD4QgkHjbBrq6zaXbiWDS0XUM28EBPBGHlrnVRCrDVE4rJxyyMLXEepSugaMSG9K0TofOalzagB22qHy3DDtqU3RGT9w5HuyQRfXkRFlNt0m9cGW7U5ZtmVtJeEkd6UcXYuZ7aX6vFFdxGpmn90Ezt3/9H6pIDc6VrQhXhNm4KU7t5jBa3Ca7m5vCAWmUfNGEWttIzzydbuf1wZW98yU9UUvx9npFSwzOGaCR708+CA11cUzLYkBFJZqjWNezO2gauuPQu7IGTyzOGk9PKEweO3/1pjgLPmtBdxxiEUWf4X7mbHTwmBoTBulYH5TgBOJ4R+ZRv6CSD3KKMUcUMGmgxqblIRrrLYYflkZufuICzFGKD0sVpU1wal+N/tmd/zA5wRiRyavBcvBOOgQ9y9vpfLusvA7e/QKKsuGMMYlbDnuAWlaGmrUWbXh3tO8cskfWKYQQB5s5c+awaNEimpqaUoaB1DSN22+/fbfWu1+D9rKyMp599lkWLVpEZWUlbrebvLw8hg8fztlnn83YsWOT5l+8eDGvv/46S5cupa6uDoCCggKGDx/OKaecwvHHH59UWzR27FjGjx/PAw88sC93S3yDrJteydTfLU1ksJUizaWIN8UA0DTV/sxlDY69bST9TujGslc2JYJo08BxtX7FlMKMxduy6QrQDB3L6yLi9eAPRvCFoknbjnkTAZ4/FMUTc0hviKE7oD6r5ZP+rxP16qh0k5ijMOIOGhqGnRjn1jEUhg22mRwMNr69mQ8HvUG8JZHhd+W4Gfuf8az88WeESxNXE65cN0d8MpGMETlty2382SyqH17euqsOpqGgNavvP7qIgR9/F93b+emk2zHd0H0GTri9442d7ua1l+soXdqCLxpLOi6WoYOmkVUf5PBfL2Tsp6eT3qEt21MXVhz7os3q+sTrTDd8dIHBEcXtx+DQ3hojp31CRjhxcRU2NKoLCjhn1gLymhP7f/Sy1SyYdCpf9ihsW85XHuLZ+6vRW6+yLE1jbm4Wtm2T+eYHFNUl+i84GlxzxdnMGpqPvsnhvHVbyG9KbMsw4KobezDi8DTOeMPhw9LEygwN/nuGzveHJl/4WJuaqB3/DPbmxLr1Aj/5s36E65D8HR6LLr2/CL57D8Ra7ziMGwwf3wm+r9ffwblvKuqml4HWz/flx6A/+5Ov11YhxLfKwdYRdVv19fWceeaZfP755yil0DSNrVXoW///6wTt++3wrFixgosvvphp06Yxbtw4brzxRq666iqOOOIIli5dyvTp09vmdRyHe+65h6uuuorFixdz4okncuONN3LzzTdzyimnUFpays0338x//vOf/bU74htIKcXsB1a3lZzottMWsKM6BOwACj6/bwXFo3M4+peHJLLeHTunaBqOYWydtX1YK03DtG2aczOJ+DwoErcHAxl+bNMgrSmELxjFE7TQO27PUhjR1gm6htMh060pwIHMxijecByUQnMUaYEYplJtATtAvCHGV9d/3hawA8TrY6z/49K21+GVDW0BO4CB0xawA4Q+q6ThhTXbPY5Gmsnm4T0Jt46eE/R7WNmvByXLgrjjyaPlaIDuJNbdlJtGpdvLhjsWb3fdHf17sWoL2AGaY3Dr7OQMR8+PV7QF7ACvjhxAen19W8AO4LZsfj3zc1yOg6EU/cJRhgdDbQE7gKkUQ1uCnLN0VVvADqAruGz2EgAG1AfaAnYA24bX/1vNlBLVFrBD4lDe9KmD7STfdwn8fW5bwA7g1IRo+ePMnToWXfrVs+0BO8C8NfDCrK+1StUcRt3+ZvK05+aiFm74WusVQny7KF1L+Xcw+dWvfsXSpUt54YUXKCkpQSnF1KlTWbNmDddccw2jR49my5Ytu73+/ZZpf+KJJ4hEIrzwwgsMHjw45f3a2tqkeV977TVOP/10br/9dtzu5Fv3P/vZz1i4cCE1NTV7vd3im61+XQuNJS0UHZaLJ8tNy5Zwa9oQtB086SwetFg9eTO9jsrbukgS2zTQbRsUuGI2lmkQdxtYhoFj6DR0y0ZTCt2yyS+vI7MhgNkax+m22nZzbcEtAI5CsxTKADQNT8jGtB2ymmNkNrfeGQACvtQOm5GaCNsWRzR/UYsdtjB8JpG1Ta1Tt1/MUzO/llLXRgjHGXpSPvEltXgOycEzPJemj8pwVQWo7pGHLxyjNj+TuMvEsG10xyGnsQV3zKI2NxPbNNBI3NUwbIeQ36RpeSMAjbUxNq4JkVvZhE+z0GIK5W4/yl/VOBCz8dk2PUJRav0eqjfarJwVo8fQDLzpJg0LqpP2dUO3XJyl61L2p6ilhWMbm4gbJm4Feif99dMtm6KmFmb264nLdjhq0xZ0YGB1PactXUPMn56yTG1VnFV1qZ+jiiDMnB+if3eT3j1MnI/X0DC7jG2LlOILylGhGJrfjWoO47y1GG1LI9rEETCqF6FPy3AiFmkn90L3bOf0vqYiddraxDTlKKIzNqJCcbx9vWiryuDIQdArH75YD5tq4IQRkLPNvpU3QCiWslq1tgptbD+cOetRtQH0dB3qW4jr6WhZPswTB6BZNny0BPwe6FMAi0phRC9YtwXWVUL/Qjhl1Ne+EyCEEHvbe++9x09/+lMuuuiitooQXdcZOHAg//73vznvvPO44YYbePHFF3dr/fstaN+0aRNZWVmdBuwA+fmJW8D19fU899xz9OjRo9OAfattS2mE2FX/z959x0lR3g8c/8zM1tvd64Vejw4CgoAiIAYrtliisUVjTYw99m40sSRqYoyJibH7s2AUsAAiKgIiIr1JkXZwve7e9pn5/bF3e7fsceIBV+D7fr1W2Wdmnnlmbm7uu89+n2cW/HE1a96I9QyqFoXux3eJB8uxHnANjYYUj6TA3IT5j67GdFgxNQ1lj55TZ20Qhz+MZpjx7WrdDqoz7WDGgnp3lZeUGj+6VUO3qBhRA1vUIGpTsUYS53U1NAVME1utjiXS0M6oTUHTTWJ7MRPaaIvqRK2JIXpT/RjBrT4W9XmPkZ+dhOfYTqhWBSJ6k+vWuJ3M/MGKua0IgEUvbmP8dxuw6AaWTinoRX4m1bcNC2XpbpaM6odqmIxZsZGcythc9mGrxuKRA6hMdWGN6qAo7O6dS7ioCu/UeczXMpi4cB3hitj6Q1Jh802xNszeFGXmshBEIQAovgjXrtqKKxzmvdmgqKB77PT2ujm2UdvHb9rOon49OH/JWhr7vH9PpvfpimYY5HqDDK5WGOxLHIy6NcXBWydPxGeP3ZMGlJTz0lsfsaFLJ47eXMiEzVt5Z/KEhG2sDoV31hns+SVndiDMi8+XkOb3ce+C97h//GS0roO4b0VigG1sKqOs11OkPz4J5Xevgj8c+wnf9T9qM3LZVZkDKFi6uen1+bnY8tMTf1ihCKQ6odybWD7lCIzqIKU/e4PId4WkshsHsZ8nmgrDe8GyH2LvXQ743+1w4oiG7fNzIcUK/oZvcbCoMLYP4Z/9FWNe/TcxBjYqsKDjoytKv1zcvh9QCivrlit1rz0+2OSlwWcPwZCWPZBECNExNDUWqyOpqqpiyJAhALjdsc4Nn6/hm9wTTzyRu+++u8X1t1l6TLdu3aiurmbevHnNrrdgwQJCoRCnnnrqXgN2IfZX6dqqeMAOoEdNfvi8JP6+/j4SsVtj/c31qS/1PbB1KSgYCmHNiqGpGKoSW9c0sQYj2AKJATuAyxfEGo5ijURQdZ2UGn/DckUhYlExgFCKFsthJxb8RqwKEbuKGjHiAXt9Oy1hE8WECCpRRcGs20rBwBGJ4AjHUmYwTSyGQcQXQbc0tFWpG0AbLgqw5c7vUFMsaHYlIe+8fo9RTeXb4fmYasOtpDLdw9ZuubHzWORPaJuGzvf9uoKi0L2wLB6wA9giOkM27ojtp1FqUVmndKqXlzNkzQ6yKxrWt9ZAl//FWnLdjDChRhkfm1JdFKQ09MyaBijVId4ZO4x3jxpKRFXRFYWjthbw2eA+/Pu4UQTrBmMuyu/Oo2dOirVfValy2tjkTqHMZokfd5HDxjeZqfGAHeD73Cye/NnRvDtuBB+PGIqi6xy7al3CNRL16wTXJwbMVsPg6B2xHplT1nzL93YP/zl6PC9OOIq3Rx+BrtR/+DJQMTBLazFueLOuZ7vhPLkqS0ipG+UcLfBRcu8ikrz2RXLA3j0bpgzH97dviXxXiEYID0UNNetGQ8AOUBuE3/07oQpl1nJUfwXEP9QaKNkaxqffNwrYAVQieFAxSKEU+6bvGwXsUPfxOLndxdVwZ/JUcEKIQ0tHT4/p0qULRUWxDg+73U5ubi4rV66ML9+1a9d+zeveZkH7FVdcgcVi4fbbb+fss8/moYceYtq0aWzdmpgDuWXLFoAme+R9Ph9VVVXxl9frTVqnvamoqCAUahhw6PP5EtodDofjX6nUKywsbPZ9UVERjafbl3389H2Ub6xJWI8mfqkUIGq3Ekhz4U9NQTVMFAMU3UQ16vsITajLZzcsGrpVQ4vopNQG432Ie7IFw6i6jjUUSV6uKATsGn67RkW2g6KuTiry7Hiz7EQ1Ba2+h11p6KFQgKJuqUTtGkGblfowW8VEBVzBMBneWnJ8vniPdtShoqgmmhl71bfDt6qCcIEPwxehcbjud9iZM2E4cyYOpzYlOW2hxtP0XPAKJl63EwCPL3lWmVRfoOmef6eN9GpfUrmzALYVFLOlIjl9pciZONOKAtTarNzxi5MZ/eB1jH7wt9x5/ilELSp/nnosRz94NeMeuIrLrzmbylRX/IcVrhsYu8WVQkhTqbRpvDu8Jz57cqrR5wN647fHzseOrEyO3rCRHsUlqKaBVvetR9YeA4413SS17hNHt8pS1nTuDEBU07j1/NOYPnIgFqJYaPimQ/X5aepqstNQd2BFSdLvR+3i9UnbUBk7r5FVsQ+pVpr+GSTYVAj+UMPv4KrtKOioeFGpRqUGtagUc/nOpE1NYudNI4RGKGn5Xq3aFv9ne76XyD5kHx1tH+LAmThxIp9++mn8/fnnn88TTzzBo48+yh/+8AeeeeYZJk+e3OL62yw95ogjjuD111/n9ddfZ9GiRcycOZOZM2cCMHLkSB544AG6detGbW0tAC6XK6mO3/zmN6xf3/BHqE+fPrzzzjutcwAtlJmZmfC+/uuTevUz6DTWue6P+N7ed+qU+HRJ2cdP30enkZnUZZPENHFDMwFTqRtgqijoVg1LeI+UFaVu2/qgX1HQbZaEOhoHRCaxp6I6fQZhqyV5uQJRuw1XbZB4n7kS6/XWjFivRNRG7IMCgGFiCRmkewNgGKg6hOoyoxVMHETqQkeFiGli1w3qQ2fdomCJJh53+rF52HuloqVa0WvC8VPkCIZwdHZS02jO88ayKpv+AG2gkFnloywrlYp0N70LShKWl6e7m0g7MknxBSnOSSevIrHe2nzo1S2PkZ2jLC9M7KHt4a3dY9+QGoqlb3gb5UdnBcKUuRz47Tb89T3nuhm/FuzR2M84OxRmt8PO3OwMiBpgVWOjSxvx1E3TqRgG/YtKCGsaZRlpCcdT6HEmttNoSCnZlNuVo/fouFjQrzfnL1uVUGZkZ6CVVbJn4B6g4cOSa2K3pN8P10mj4cUvEsqYMBgA24TuBKatJ4wLEwWluQlJj+wDKXZsENvHhEGx4wbiJ+7IPqjH9Uf/18KETVViue9RnBhY0IiwT44dFP9ne76XyD5kHx1tH+2J0bE61pPccsstfPrpp4RCIex2Ow8++CBr166NzxYzceJEnn322RbX36aT6+Tn5/Pggw8yZ84cZs6cyYMPPsjIkSNZvnw5t956K5FIJB6s1wfvjd1xxx0899xzPPfcc0kXrRA/RXovN8fcPgSLI5YikZJlZ+SlvbCmxN7b06z0ODanYQPDRFfVWN4usTBFt2gYNktsfvVGKRG6qhCyWzEVBV1tCIVMIOiwYGixOgyrhdoMN0ZdwG8oCrWpLtx1vfRQFxQpkOLWsESM2G9w468PVQXFYpJWHcCa40yYccZEIYQFGtcWMck7JgcUiFpVlBx7fLFnVBb5T4wmtKUaoyac0AaL28oZd+TjzrTGZqtpdLxdi8rpsbsUxa5iH5EdP14DBR2NPluLSc+0sDsvkx1ds+PfEHjdTtYM6IEzVcPmiJ0TRTfosquCTuf1YceoXmzrlhOvz98ddp0b2/jFs230rJvrXDVNRpdV0RkjlosPODwW1E4u+lX4GF5YgVLf3ohORmktQ4uq4wN7LVEDgrEPI7aoTkYgzJgsg0E+L19lphHSVPCGwaHFf/4AOaEwncJRbJEoly5cgiMS4YGzp7A7zR3/eQ8dn8b4CZ74uTwiB/5zppXMzNh1NnfEUeR0dvPIhzNxRGLB7ILxw6g+d0T8g6DliDy0t6+C7pnEA2RVwTtwAMG6oN05rhO5jx5DknPGwdUnQH0606Bu8GxsWkb3taNwnjsIAyuVdMdQ68Y+ZLrh3KOhfvrSnjnw4nWJ9U4cAnee3bBOr1x48TrUXxyJduUx8WtUIYqFGnSs1JJLZMo4zNF9G+qxWAA1+Zuukb3hyV8lH48Q4pDS0dNjhg0bxi233IK97hvXjIwM5s6dS0VFBdXV1XzxxRdJH6J+CsVsZ9+RmKbJlVdeycqVK/nPf/7Djh07ePjhh7nqqqu45ppr9rrd6aefjtPpTOhpl3naxU8Vqong3eUno68HzaYSro1SvbOWjF5ulv7je5a/uBlTVVGM2BNMFRWm/mscilUle2Aab126mOqCAKZpouo6pqJQk5mOqhuxJ6GaJhGrBXeND7s/mNBPWpvmRrfGkiAsuk7UYiHFFyCztCqpnWd+eQpKSGfZxV/hW5e4XNMNUkIRLAMz8G3Ys8fbxEFi73DPPxxJ7mX9ifqieAamEdxZi+6L4BqUDkDFaxvYfuncpDYMWn8R1vw0SrcFSMuzU77JC4EoXY7KILSuEmtPD0WlOq9cuwprVCdot+IMhqlxp3DBfX3I6WzHmWrBEQoTrQhBjzSqS0Pk9krB0E1Kt9ai1YRI6+PGnuUgEjYoKQiRqofBH+L/vvkAgMsvvxyr1YpumKwpNshxgFEZJruzDQyTqsIQ2T2cvDa7lnc+qCKqKPhsVrbaLVTUPYvKapg4MDEcGldu2IHPaiGsqlgNA4dF4em3h7C1MEKfZ4KJJ0FTeCC9jOMenQWqydTfX45iGvQsr2RrTiYBm5VHxsMEV5QjummkZ8ZSQ3Z5TapCMCQ7dgXouknBrjCZGRY8Hg1jSymVIYWd2VkMyQKrpqDvqsGsCmAZEps/3jRNzJU7obwWZXg3lGwP4e01mIEo9oGJvXNJdldAhTc2sHOPADm6vQozEMXa1QWbC2OBvcMGpdWx7Yb2iE0635S9rGPursKs8KNmOaDMi25zgs2C1ruunRsKwGmDnDT4fjf07ww7y2KpO+nuWBuEEIe8aXlvJZWdW3xBG7SkZdatW8fgwYMPWv3t7omoiqIwdOhQVq5cSUlJCcceeyx2u52PP/6Yyy+/XAajioPKnmrFntrwNEyby0LOwNj7nCHpsUC90dSP2YPT6XZMbvx9Tj8PtT/UoJhmvEdYMQwMiwaN5p4xFAXdomGJNgTQWjSKbrWAqhDVYsFdpIm8aXuWHVeXFDSbiuZI/rJMq2tf6qispKDd6raAb4+UjlHZOLs1pJ85uiemojlH5bInLcuBrZcH1aLSKT+2frcjGx6A5BwR+1Yi266jZzgJBGNtCtusKCp0yU8hI7c+RcWGtUusN9qZGrslqZpC5wEewNPQdptK1z5OwEkkEoFv9miTqjC8c12gmNGQgtIpP1bngB5WApbYv+26QXooSmVd6lJEVYigcExGLO8/NdKQ9tOrX+z4euZZ6JmhsL2yUT+HbnLicRnk3luLHjXxBAIUp3tY1+jBTFN6aYztnHir7epR6NpwaGiaQs8eDSk7at8csoDG3x9qXVOha2r8vaIoKCMSZ1Ox9Uxln3TJjL2aYOmZ3vBmZJ+Gf+ekxV7N2cs6Spd0lC519XbOTJpilIGNgvKRvZPLhBCHhY4+e8zQoUMZOnQoF1xwAb/4xS/Iz88/oPW3WXrM4sWLiUaT82GDwSCLFy8GYjnqmZmZXHrppezatYuHH36YcDh5LmBABlWIg673lM7kn9I1/t6RaWPCvcMS1olWBeOpF/UDTz3V3oT0Eae3Fqc/SNhhi6XY1LGEomCYqEZDek3EbiXaOwOlbjXNqTHuj6PQbCqVS0qpXtboaUJ19dsiOplTuzPw+aPJPrWhvfZuKQz6xzgsmQ3BYd7l/cg4ufngyDk4k073HdWQ4uC00P2fx+31CaiNOVI0TvtNdyx1aSqqBif+qkujgL31jBtm56SjG4L5fLvJ+J4N4WP3dJW//yqNIUc3BJ2eDAtnXB07h6qq8MLZDlLrmq4o8LtjrBwzJo3spyajWRQeeW8urlDDvPg3j1IY27mD/xUSQohWYipK0qsjef7558nJyeH+++9nwIABjBo1iieffJLt27cfkPrbLD3mF7/4BdXV1UycOJH8/HwcDgfFxcXMmjWLHTt2MHXqVB566CEg9kTUxx9/nPfee4+8vDymTJlCz549ASgpKWH+/Pls3LiRCRMm8PTTT8f3Iekx4mCo2OzFXxak08hMLPbEPsP/HD2LSG3ih1ElaqBrKlGrBUskilb3K+dLdWGqKpZwFEMFw2pFr5t5Jv5LqSgceUYnxp/XmZofvGSPyMKWGut93/KXtWy4P/lpoeM/PYH0Yxp6emtWVBCtDJF+bB6qVUX3R6lZWIy9h4uUAen7fNzhHV6CGypJOSoXS4bjxzdopLY6yu7NfvJ6OkjN3v9vyyKRCC+99BLQkB6zr7YXRSmt1BnW14bdprCyIEplwGR8HwtWLfYHomh7AF9VlF6DXVisiX0b3pDJ19t1+map9M1qWBYt8hFaWUpocC5LdSf56Qp90jvWHxwhhGhL73R+O6nsF4Xnt0FL9k9xcTHvvvsu77zzDgsXxgbjjxkzhgsuuIDzzjuPLl26tKjeNgvaFy9ezJdffsmKFSsoKSnB5/PhdrvJz8/n1FNP5fTTT0dVE/9Yfvfdd7z//vusXLmS8vJyFEUhOzubIUOGcOKJJzJp0qSE+S8laBet7b2LFlCyuiqhTA1HMbXEwXUmELbbcPmC2EIRDFXBl+rC73HWpdI0yO3l5MK/DcfhSezZLv64gKXnfZFQ5uzhYvK6s/ZrHtiOYH+CdiGEEO3TW12Sg/YLdne8oL2xXbt2xQP4JUuWoChKLMWzBdosp33cuHGMGzfuJ20zatQoRo0atc/rL1269Kc2S4j9cszvB/PRb5fEe9sVw0CLGrEZBBs9iVQBPFU+LHVPXFUNk9QqH05/kLJOmbEgH8A0Kdvs44vntnDynQMS9pV7clfyTu9O8czYXNiqXWXwk6MP+YBdCCHEoamjzRazLzp37syQIUMYNGgQa9asaXI2xH3V7gaiCtGRdR6ZySWzj2fHwlKcGXZSuzooXlqBrzzEN89saFjRNNH05C+5IlYL9kCIkMNG3YMwUYCt31QkrauoCqPfmkTF1yUEtteSPbkT9jxn0npCCCGEaD2mafLFF1/w9ttv8/7771NWVkZGRgYXXHAB55/f8m8OJGgX4gCzp9ro12jAamp3d13KTKOgXVFiAzuNxMA94HYSTLHHZpVptCity95zyDOPzoWjD1DjhRBCiDbS0WeP+eqrr3jnnXeYNm0aJSUlpKamctZZZ3H++eczZcoULJb9C7slaBeiFeQOSyf/1K5s/nhXvKz7Gd3ZNXMnZl2Pe9bwDKJds6nd6EfVY3O6A1jsKhOu7N0m7RZCCCFaS0ebLWZPkyZNwu12c/rpp3P++edz8sknH9CpyiVoF6KV/OzxkQw4sxsVm7x0HpVF7rB0am4cRMG8IlydnXQ/oQsm8P2iSqqKglhME4tVod+EbDxtMEWiEEIIIfbdu+++y9SpU3E4ftoMa/tKgnYhWomiKHQfn0v38Q0PK0rt7WHwFZ6E9QZPytpzUyGEEOKQZ3TsjnbOOeecg1q/BO1CCCGEEKLNdfT0mIOtzZ6IKoQQQgghhNg30tMuhBBCCCHaXEefPeZgk6BdCCGEEEK0OUPSY5ol6TFCCCGEEEK0cxK0CyGEEEKINmcqya+Opqamhscee4yTTjqJkSNHsmTJEgAqKip46qmn2Lx5c4vrlvQYIYQQQgjR5jr67DEFBQVMmjSJnTt30q9fPzZs2IDP5wMgMzOTf/3rX2zfvp2//vWvLapfgnYhhBBCCCH202233YbX62XFihXk5uaSm5ubsPyss87iww8/bHH9kh4jhBBCCCHanKkoSa+OZM6cOdxwww0MHjwYpYm29+nTh507d7a4fulpF0IIIYQQba4j5rA3FggEyMnJ2etyr9e7X/VLT7sQQgghhBD7afDgwcyfP3+vyz/44ANGjhzZ4volaBdCCCGEEG3OVJWkV0dy00038dZbb/H4449TXV0NgGEYbN68mUsuuYSvv/6am2++ucX1S3qMEEIIIYRocx0th31PF198Mdu3b+fee+/lnnvuAeDkk0/GNE1UVeWPf/wjZ511Vovrl6BdCJHku9UB5i2sxWpVOHWym/597G3dJCGEEKLdu+eee7jkkkt477332Lx5M4Zh0LdvX84++2z69OmzX3VL0C6ESLDgWz9P/LM8/n7+N34evyuXfr0lcBdCCHHwdLR0mMb8fj8TJkzgqquu4tprr92vNJi9kZx2IQQApmmy+esK3nizDGckSkYgSEo4QjRq8v4nNW3dPCGEEIc6RUl+dRApKSls3bq1yakeDxQJ2oUQAHz0+Cam3b0OvaCWrr5asoIhutT6yasNsPIbL1s3+Nu6iUIIIUS7dfLJJzN79uyDVr8E7UIcJgzDZOM6Pz9sDCSUm4bJ+k92s2Z2Cbqi4IpGE5Z7IhE8gRCz3y1tzeYKIYQ4zHT02WPuu+8+Nm7cyCWXXMKCBQvYtWsXFRUVSa+Wkpx2IQ4DFWUR/vrITkoKIwD0yndwwz3d0L0RZlyzmLKdIUh1Yahqk1/tOaI6lWWR1m62EEKIw0hHnz1myJAhAKxbt44333xzr+vput6i+iVoF+IwMPOdsnjADrBtc5BZH5STvqmI8m1+/A47UVXBousohoGpNvoSzjSxRaMMGZ1NyBtBtahYnVobHIUQQgjRft1///0HNaddgnYhDgNb1tYmlc2ZXoFNtbBl5BAURSWn1k/3ikpcwSB+hwNDVQmrKmVOO+5wmPIvdvPi08tRrSpDz+vB+FsHHtSbkxBCiMOLqXTsrO0HH3zwoNbfsc+OEOJHhf1RogXepHIFiBoq6ZEougJep53izHRMVaXSbmNzeio70jz4bTZK3G6WVtqIKip6yGDl69vYMGNX6x+MEEKIQ1ZHz2k/2KSnXYhDXOGaGjLKqqjNsxK2WutKzfjy1HAERzhCls+HMxLFl+LEqRt08fmpsVnxW60YChSkeSgc0o/uZZX0LK3grQ+9lH22nRKHnfWmhWKLRnaqyo2THFx7zP7N6b6jMMJ//uflh4IIg/rYuPrcVHIyJCVHCCFE+/Xwww//6DqKonDfffe1qH4J2oU4xKV1daJikhIOAxCxWmjcd6GaBr3KK1FNk5KMdFAULKaJRdeJRFV8dhsQGyAU0lQ2dc2jJDMNewC2aDZ2mRpbrRroUFZq8JtpftIcCr880tai9oYjJnc8XU5ZlQFAcXmAHYUR/nV/7v6cBiGEEO1cRx+I2lx6jKIomKbZfoL2goICXnnlFZYtW0ZRURE2m42srCyGDBnC6aefzujRoxPWX758Oe+99x6rVq2ivDz2BMacnByGDBnCCSecwKRJk/aaM/urX/2KtWvXcsYZZ3D//fc3uc6DDz7Ihx9+yNy5c0lPTz+QhypEu7P+m2pWza/E4dI4+rRscns4WftlORsWlKN1cWMGFZzhMM5QCKtuYCoQ0SyopomhaQQ1NelBFu5IhFLTES+3GCbucAhQMDDJDIXZaE2+jTw6y8/Cr/1UBQzSrCZDC0oZs6sYfVI33uncmfUlBrpukhLR6RLS6Z6m8LNxKQzxRJn9ly2UBXMS6vuhIMr03y2jiy1Kn0v6kjo0Nb7MMEz+vSTM7I1R8nZUct7KH+jZP4VuNw5Bc6hU/G05we+KMUd24r9HDCb89Q7OXraC/lVFON1BLJPyYWx/+L+vY8d51fFwTP+E/ZuhKMHnvyG6YBvasE44bjgGdeFaePMrSEuB350CQ3ocoJ/kj6sMmvxtqcH8nQa1YeibBleM0Di+l8qWKpO/fmdQ7Idz+in8YmDLsiDXlJo8t8KgJgwXDVI4tY9kUwohDrKOHbNjGEaTZdu3b+e5555j/vz5fPLJJy2uXzFN0/zx1X7cunXruPrqq7FYLEydOpU+ffoQCoXYuXMnixcvZty4cdxxxx3xA3jiiSeYNm0aeXl5TJkyhR49eqCqKrt372bRokV8//33XHfddVx++eVJ+9q8eTMXXHAB3bp1o6KigtmzZ+N0OpPWk6BdHC6+nV3GB8/ujL+3OVWOOSmDxe/sRotEsfmDVGRnohkGjnAYBQhbLEQtjQJu08TnsCcE7roC21M9jYJ2g4xQOGGdgKrwXlY64UYzzqjAwFCExumIR6/exv8Gdqfc3fC72isUZkCoYVabs5ZtIGVXDa+dMTbpGM+fu4TUQAjVqnLs2xOYsXoGAKvzLubvixumz+pUVctLL84mK99DblaEwIKG3Psfcl2MKdnaaDCPQSoFWBqlC6GpMPceOG5wvMh73puEp61pWKWHh7Qd3zb8fXE5YNmfoX+XpHYfaIZpMvqVKMuLEwpRDPjPVI3bFphUBBsWPTlJ5fdH/bSAe325yVGv69Q2muXzlVNULh0igbsQ4uB5dnTyg4muX3pSG7Tk4LjoooswTbPZ6SCbc8B62v/9738TDAZ588036d+/f9LysrKyhHWnTZvGKaecwn333YfNlvg1+nXXXcfSpUspLW36YS7Tp0/H5XLxhz/8gcsvv5xPP/2UM84440AdihAdzoL/lSS8DwcMvplZgi0YwhoMEbZYsESjpASCqKZJxGohqu2RI64oWKNRIvG8d6i22UBR8NQGyCurRLFoVKZ7QNNiAatpklMb5LRQiEWZ6QTsdqyYhFSVgKrgNE1shkF2KMKCQd0odyV+uN5ps9IvFIkH0Yu7deaSZTvoWlTJrk4Z8fX67yiiU5kXTAjZLfzwyhbcHoP0hRAuns29bie7MjxETYVF+d154tSjuGTxWo5enzhYNr+6ZI/R9yoBMvFQjgmEcKPrVuz3vIflnM2Q4UIfNyQhYAfQd3iJkoKVuqfE1gbhtlfgmAHQOw+2lUAwDJoGR/aBk0fu9+O4V5cYfLLZwBuB5UVmYn2qgmmY3PWFTkU08Qj/9I2B2wqFtSaY0DdD4dz+CinW2PaVQZO3N5hEDDhvgEInl8ILq4yEgB3gzvkGHhuc3lfBUvdpbFWJwawtJr3S4OcDVHb54H+bTDLssbrctgPbbRY1TKZvNNlcaXJSH4URefIhQohDSUdPj/kxEydOjHdgt8QBC9p37NhBWlpakwE7QHZ2NgAVFRW8+uqrdO3atcmAvd6eqTT1IpEIn3zyCccffzzDhg1jwIABTJ8+XYJ2cdiqLAlTVhhKXuAN4fQHKcrNRjUMcssq4j3DtpBCwOFI2iSgWQharWimQVRVUUyToVt20KegOL5tjSuFVUP6YioK3csrcUYidAJGlVWypHMumzLSMPRYz7WBgj1q0MsfwOdK/jbMoPGQWIhaVBTggk++Y/mg7hRnecgvLmHM99vQDBM/NqI+DeWjrQwsiPWud2I3KgaKYnLVBaewJS+DLXkZfD64B/fOXsBFy9YB8OeJo3lwwdwm2mDBRKGSrkRxYsVHyqIVsGg5AGp2Kgp5mCR+yDH3/B53xrexV1N+OQHevLnpZfvgP8t1rv4o2nCuFEAj6YNASa0Je4wBrgjCb+Y2/srW5LFv4OuLNCoCcPSbOsV1nz3uXQBfnK8lBewAhbVw9nSDE3oqzDpX5b8rDa7+WI+3aUiuwWavQqjuS48/fgOLL9LIch6YP8K6YXLCm1G+2BHb452fw3Mnafx2lAxQFuJQcajPFrN06VJUteWdDQcsaO/WrRvbt29n3rx5HH/88Xtdb8GCBYRCIU499dS9BuzN+fLLL6mqquK0004D4PTTT+fPf/4z27Zto1evXi1tvhAd1vzppeimgkZippszEKQ2xUnQbqNLcSkhqwW/3Y7HH8BqGERUFWuj7DgDqHY5Cdf3ogNaNJoQsAOk1vrJLq+mNDsdS6OnuinAkNIKvk9PS8hLrLZZKbdayIxEsRgm0UY35U6RaEIoPHhzIQC2qM7Y1duwESWFcHy5nSgGKraC6oRjNVBZ0Lcz3/bqnFD+t4lHcd6KDexOc5MeDKMbNmhUH4CORgg3UWIfKtwUozQ6l0pZDSmo1NIpXqYSaehl3xf/9xXcdiaM7LPv29S3zzC5+/No4k/XrHvVn8r6n+M+/sFbXwH/XW2ypcqMB+wANWF4ZLHBhK573/bT7Saztprc/YWe0Ka1FSRMIry5Cl5YZXLX2APzR/jjLWY8YK9333ydK0eo2LRD+w+9EKJjePXVV5ssr6qqYv78+fzvf//jyiuvbHH9B+y7xSuuuAKLxcLtt9/O2WefzUMPPcS0adPYunVrwnpbtmwBaLJH3ufzUVVVFX95vclzS8+YMYMuXbpw5JFHAnDyySdjsViYMWPGgTqUg6qiooJQqKFX1OfzJRxnOByOD8qtV1hY2Oz7oqIiGg9NkH0cXvuoKAljqAq6Ggs1DUBVTWyRKBVpHsIWjYKMdL4e2I8VfXqycHB/StI8VDodVDvshDUVv9VCUaoHXVUxlIbeb0NVmxwX5AjH8tr3TLFRMZtMA9ngdvGDx03XSJTMSJRcX4Bzlm7m97OW0bm4ipxyL0d/twW3P4CFKCoGdsI49wiwVUyU5D5uAHaneZLKapx2ClJT2ZaeRtdqLwFS8OImgoUIFsrVNEJ4qCW90T6Su5ntVGOnCpUQNmqAWsroTpAUwnt2be9FYP32+L9/ys/cH4HSpj4fGGbspZsQNUE3wLLvwev2GpPvS4JJ5duqTVw/ktaysdJMblMTm2yrNg/Y78e2quThVxUB8Iba/ndQ9iH76Mj7aE9MRUl6dSSXXXZZk6+bbrqJ+fPnc+edd/K3v/2txfUfsIGoEBsg+vrrr7No0SIqKiri5SNHjuSBBx6gW7duPPLII3zwwQf84x//YMyYMQnbX3LJJaxfvz7+vk+fPrzzzjvx90VFRZxxxhlcccUVXHPNNfHy3//+96xevZqPPvoIS6OBdTIQVRwOFs8u53/PJ+Zu9+upUfFdKTu75BFRFEpcroaFpok9EiEjEMTrSknYLqoo6EpsHQ0Flz/A2JUb0IzE28SyYf2ocKcwZFdRQk95SYqDWb26kxDBmYmBvGaajNpWzIWzlifUuTq/M2sGdObaTxbgiEbrOpMTb9hhNIJYSbVHiOdhxHZCYUYKZ1x9Lkajrx77llXxyIcLCWkqHw3uwcNzFibUpyoRPGYAMOs+EICHXTipZG90NDYzJp4uYyVAH75rftIDlwN2vgAZ7ubW2quJr4T5amejn4FZF6zvyaGBZd/6Ymafq7KlCn47N3G2g/vGKVwxTCX/RZ1o8kQIWFTYdIXGpTOiiW3SlNirkelnqZyRf2D6hjZVmAz8V4TGl+KYLgrfXGbd+0ZCiA7l6aM/Syq7+euftUFLWmb79u1JZYqikJGRgceT3LH0Ux3QUTz5+fk8+OCDzJkzh5kzZ/Lggw8ycuRIli9fzq233kokEsFVFzzU1iY/Vv2OO+7gueee47nnniMrKytp+YcffohhGAwfPpydO3fGX6NHj6a8vJyFCxcmbSPEoW7MCZlMOCMbq01BsyiMmpzBRX/oj2d4bBxJpFFvuDsYYkhRMUOLS8nx1eIIR2IBoGmiAxZDxxOJ4onq2KNRohaNgrxsytJjN5uw1cKm3t2ocbvYrmn4rNZ4r3yV3caiLnmxVOu6vgDVTO55t5gmm3vk8NmYfgRtFgwFNvTMYdHkgUy9oRcLJw/Bb7WiALqiENZivf9Ru5UgVtInd6HHuycSzK6v0cSCQefqWh76cAEZtQEA0h3w72N0HDkO7LrB0dkK0yYNp9ZqIaoqVDlt2FM1lC4eqOu/NwEfnQjndIrlVqamwG1nwUkjQFEwrRpechPy2yM42c0AjKy0WEFqXe5+ffDcJw+m/b7FATvAa2damdgjdh4zHIDRdF9LZ6tB3Y89nimT4wS71vAxKs0GT0xUObGXyjXDFW4epeC0gFWFy4cq3D1OpWeawptTVbrV/Y1x18XFXd3wxqkqvdIUXj3dwsTusVo7ueC5KQo/76egKpBqgz+MP3ABO0C/TIVXT9foUncax3dTePNMedSIEKL9UBSF3NxcevbsGX/16NEjHrAHAgF27NjR8voPZE97U0zT5Morr2TlypX85z//YceOHTz88MNcddVVCb3lezr99NNxOp3xnnbTNDnrrLPYtWvvj06fNGkSf/nLX+LvpaddHE6iEQPTBKstFihtXu/nmQe2x3raU1JQTJMjCouwNAr4DEVhd3oaEU0laLXg0BO7VqOKQkRVCWgammGgK0oseAUGFhbj1vV4PnVEU/m0dw/0+pQa00QzTao0DWOPnva0urlsFcNAM0zSs6y884ds7FaVx/6vhve+rMUZihBw2FANk8umOLn27FSMoI7mshKJRHjppZdQAyYXXXIxt88x+fsaBWc4QqBu3vj7Jtt4eIoN0zDRgzqWlFi51xfFZSE2TaJdQ9FUDF8oFmSbsakTFZcNAiGwWsBSF6D7Q2C3oBfXsrnfy5j+aMOJsqrkb/01lnRbrFe9NghOW2wbl2O/Z46pVxs2qQyY9P5rKKEX3GWF739np7NHQVUV/BETm2oS1GMzuIR1EyP22QyrRnz2l3ph3cQ0wb5Heo1hmvgj4LYp+MImKVZQ9ziW2rCJs1F5IGI2uY8DxTBNAhF+NIVHCNHxPHXMvKSyWxbtfZxke6NpGq+99hoXXnhhk8vffvttLrzwQnRdb3L5jzno3RSKojB06FBWrlxJSUkJxx57LHa7nY8//pjLL798nwejLl26lF27dvHLX/6S4cOHJy2fPXs28+fPp7y8vMleeiEOdRZrYq9m/qAUJp+ayRefVGDTdex1A0EbU+tSZYIWR9Ky+uUAVjM2cLU+YC82dUbX33TqgjWrYZJXG6DQE/s2bcIRdjTT5PPVYbyqiqnEemEvOT6FdM3kjTm16Kg4XAp3XZKGva79V011sWprhI0FsfdD+tm45LRUFE1FcyUeo+FUsHhs3DNV46uyECuLYl3Cx/ZUufXY2L8VVYkH7AAed/JtT3U3kZfu3KMsJfZe65JKp3/8jKLffIYZiKLYNHKfmoSla8MDn3DVzczjTp4xZ3+4bAoum8JTJ1n5/ZwIYR2cFvjHVCtd0xrOTWw6RwV33eeNHxuoubflqqLgrrtF7236xj2DZ6f14AbTqqLgatnDdoUQ7VxHy2Hf04/1g0cikfYxe8zixYsZPXp0Qk45QDAYZPHixUAsRz0zM5NLL72Uf//73zz88MPcf//9TQbuex749OnT0TSNX//612RkZCStn5GRweeff85HH33EpZdeeqAOS4gO7ZzL8pg8NZP7b9tKTd1A1ca3RJNY+kxaOILfYsHQEm8m9TdQi2mimToGCiomE/vbYTexUa+NdAqGSDFNrr+5M+NGxfLltxVGKK0yiKrQM8dCl8xYJPnziSkUlEYZ1NOK096w36xUjTfuymTd9iiaBgO7/3jOciePyvLrHHy7y8CqwsguB3cawLRfDcF9eh+CK0uxD83GkpPy4xsdQNePtXD+EI21pQYjOqlkHKBpFYUQQvw0NTU1VFVVxd+Xl5c3mQJTVVXFW2+9RefOnZOW7asDFrQ/9dRTVFdXM3HiRPLz83E4HBQXFzNr1ix27NjB1KlTyc/PB+Cqq66ioqKC9957j+XLlzNlyhR69uwJQElJCfPnz6eoqIgJEyYA4PV6+fzzzxkxYkSTATvEBrtmZmYyY8YMCdqFaCRcEyatpIaqrAwqU5xk+gPxZVUpTgJ2G9ZQGLseJahYsEd1NNMkrKnU2O0YmorFMNAADZPUVJVbrsxhdU6Exe82zErgs9vRLRpTjnHFA3aAXp2t9GriHpWboZGb0XRwrSgKQ3r9tAGGiqIwplvrzdmtZTpxTe7RavvbU65bIdfdescrhBAHW0fsaX/66ad5+OGHgdjfoZtuuombbrqpyXVN0+SRRx5p8b4OWNB+yy238OWXX7JixQrmzZuHz+fD7XaTn5/Pr371K04//fT4uqqqctddd3HiiSfy/vvvM2/ePMrLy1EUhezsbIYMGcLVV1/NpEmTAPjkk08IhUJMnjx5r/tXVZVJkybx/vvvs3LlyiZTaIQ4HFkdGnZdp09pObvT0/DZ7TiiEUJWKwGbDa0+j92E1GAoPjrdputEFIVSVwoRRWH8MBujR6QwdowLt0sj7+qe9D8mg13rfBgOCz7FQveedgYNa91eZyGEEIeGjhi0n3jiibjdbkzT5Pbbb+eXv/xlfFryeoqi4HK5GDVq1F4fHrovDvpAVCFE23vrge/ZuLiKkKZRkJnRcGM0TVzhCFbDwFDAEU0cHGMAWzLSMBWFkUPsPHRLbus3vgn1A1EBLr/8cqxWmfZPCCE6uicnfJlUdttXk9qgJS3z0EMPcc455zB06NCDUr/MlyXEYeDce/qx9MNiCtb7GJNjo1y1s3FDkNJdQRQgYLGgmiYOEoN2hdj0jaaiEArL53shhBAHj3mQZp1qLQ888MBBrV+CdiEOAxabyrizExPLN38f4MmHdhLRVCJ10zQa4cSHN/gtFjTTxDBNjjvahRBCCCGat3DhQpYtW0Z1dTWGkThjg6Io3HfffS2qV4J2IQ5T+QOcXHtzZ6ZPq2DbrtgTSKsdDlyRMJoRG4ga1VSyQmEMoHfOAX0WmxBCCJGgI+a0N1ZRUcHUqVNZsmQJpmmiKEp8NsT6f+9P0C5/hYU4jI08ysO9j/YgvW4WF11TqXE4qEhxErZYUOpuoCrw+ovFbdhSIYQQhzpTUZJeHcltt93GqlWrePPNN/nhhx8wTZPZs2ezceNGrr32WkaMGMHu3btbXL8E7UIc5iwWhetv7ESXLrHBnFnZFvL72NjzVlm8O4Lf37KnuAkhhBCHuo8//phrrrmG888/H4/HA8RmN8zPz+e5556jV69ee50Ocl9IeowQgr75Dv74RA98Pp2UFJWN6wI89WhBwjq5naw4nfI5XwghxMHR0XrW91RVVcWQIUMAcLvdAPh8vvjyE088kbvvvrvF9ctfYCFEnNutoaoKA4emMGlKWrw8xaVy8RV58XQZIYQQ4kDr6OkxXbp0oaioCAC73U5ubi4rV66ML9+1a9d+/R2VnnYhRJMuuiKPKadmUF4aoW9/J3aHfMYXQggh9mbixIl8+umn3HPPPQCcf/75PPHEE2iahmEYPPPMM5x00kktrl+CdiHEXuV1tpHX2dbWzRBCCHEY6Gg963u65ZZb+PTTTwmFQtjtdh588EHWrl0bny1m4sSJPPvssy2uX4J2IYQQQgjR5jp60D5s2DCGDRsWf5+RkcHcuXOpqqpC07T44NSWkqBdCCGEEEKIgyQ9Pf2A1CNJqkIIIYQQos2ZSvKro9mxYwfXXnstAwYMIDMzk/nz5wNQVlbGDTfcwPLly1tct/S0CyGEEEKINtfR02PWrVvHhAkTMAyDsWPHsnnzZqLRKADZ2dksWLCA2tpaXnzxxRbVL0G7EEIIIYQQ++n2228nPT2dxYsXoygKubm5CcunTp3K22+/3eL6JT1GCCGEEEK0uY4+T/v8+fP5zW9+Q05OTpPzsffo0YNdu3a1uH7paRdCCCGEEG3O6GBB+p4MwyAlJWWvy0tLS7Hb7S2uX3rahRBCCCGE2E9HHnkkH330UZPLotEob731FuPGjWtx/RK0CyEOmOotXr59eAWL7lhK0dclbd0cIYQQHYiJkvTqSO666y5mzZrFb37zG9asWQNAcXExc+fO5cQTT2T9+vXceeedLa5f0mOEEAdE2epKPr7gS4zaKLpVY+O07Rz31zH0Oq17WzdNCCFEB9DRctj3dMopp/Dyyy9z44038sILLwBw8cUXY5omqampvPrqq0ycOLHF9UvQLoTYb5s/3sW8O5dhoGK6bETtVnSrhS+e+p7LJGgXQghxmLjkkks4++yzmTNnDps3b8YwDPr27ctJJ50kT0QVQrStcG2ULx9YgaGbACiAJRRB1zSqamHXiiq6jkhv0zYKIYRo/zpiT/vdd9/NBRdcwBFHHBEvc7lc/PznPz/g+5KcdiHEfqnc4iXi1xPKFEAzDAA2zi36SfWVbPOz5IMitiytwjTNA9VMIYQQ7VxHnPLxsccei+evA5SXl6NpGvPmzTvg+5KediHEfnFn2VAME1NtuLmagKEqKIC3MLDPdX07o4hZf98Wr2PA0emc/9DAA9peIYQQ4mA6WB1O0tMuhNgv9nQbmcV+FKPuJmWauAIRUGO3F2eqdZ/qiYQMPv/vTgxFocbppNzj4evVUaa/2PIHUQghhOg4TCX5JRpIT7sQYr9YXFZya/x0rvHhd1hxhCLYogZF/hDbu2XRY2jTA28M3WTlnFJ2rKkht3cK+WPSCdVGMRQVlz+AZrcRcDiY90k1w8Zn0Gfg3h9YIYQQouPr6A9XOtgkaBdC7Bd/eQgNA4tukFobAiCqqbi9QfpsKaLz6PGYhknlqkrsmTZcPdz4qqP874+b2L68Ol7Pui/LcUQjEDXreuuD1Lh1atwu1iz1StAuhBCiXdq2bRvLli0DoLo69ndt06ZNpKenN7n+kUce2aL9SNAuhNgv6z/cTcBhxVMXsJdkuynonI6pqqi6wQ+zd1P4n++p3V4LCljHdmaJ4cYeiiQ8NmP397VYdbAYRrw81evD63SSkbVvKTZCCCE6ro4w8LQp9913H/fdd19C2W9/+9uk9UzTRFEUdF1PWrYvJGgXQuyXsC9KQZd0BmwqJuiwsrNLBtTdeA1N5et/biF7Vy2lqSk8e8pR7MhOpU+ll1+s3dZEbSaqYaDWzTyjqCpdaqsJ0oUHHinENGHKZA9Hj7W33gEKIYRoFR0xaH/ppZdabV8StAsh9ku/Ezux/PkNrO+bTcRhjwfs9QwDQjaNuy+cTHWKA4A1nbIYv6OErl5/fD0TiGgWnKFwbDtFIWizYYYN5j+1kZ3ZmXgCId5eXkbR+BQi/hSsebHtq8vDbPi2Bne6hYGj09AsHe/GL4QQouP51a9+1Wr7avOgvaamhlNOOYVQKMRDDz3E1KlTm1zP5/Px1ltv8fnnn7Nz5050XadLly4ce+yxXHLJJWRmZiasv3v3bs444wzOO+887rjjjtY4FCHaHdMwQQFlz0BaN1G1RlM06gaKFpvtxTRNMEFR9y3wDVUEUXWDiM1K2GFFjRoJaS+YJhs7ZcYD9npvHtGXs9dto0+lF1NR0FUNVA38fsIWCxUZaZh1M9Cous6gXUWACarK1plVQE9sPavZNMLLG4/tQI/G6u3a18HVf8zH7rQ0eazxc2CYqD9yjPuyjjjE6DpoWlu3QojDkgxEbV6bB+2ffPIJ4XCYrl27MmPGjCaD9u3bt3P99ddTWFjI5MmTOfPMM7FYLKxevZr/+7//Y+bMmTzzzDMMHTq0DY5AiPbHiBp89cxG1k4vQFEVhp3djfHX92f9h7v5+p+b8ZeH6DU+h2Mv7s7uWxbh/bIQe99UQif0YtXaWvSwwYCpXZh0x2A0295nhjW8Ib67cQEKKiaAosTmazdMFGK951k1tVRmWIkX1PE6bHzVqzPdvcF4mVI3t63X7YoH7ACGplGb4sQejYJpouk6ChDensZb923CEwpjKAq1Djtbtlu457KNjOiv4V9Tjrc0RK+xmZx050DcOXa8NVFef6GYVd/VkpquccZ52Yw/Pi3huPx+gxdeLmfJd35cKSpnn5HGKSek7vfPRbRjz8yER9+Dqlo492j417WQKoOfhWhNMsVj89o8aJ8+fTqjR49m0qRJ/OUvf6GgoIBu3brFlweDQW6++WZKSkp4+umnOfbYY+PLzj77bM477zx++9vfcsstt/DWW28l9bgLcTj67tVtrHx7R/z9ste3U7CghJJtDQ862vpVKeVzChiwthAFCG2pwfhhFebgrug2C+veL6BwRSVZfdw40q30O7Ez6qBM/rXSoLAWft5PYexjs6ksi4LVBqaJaZoYVi0WtBsmuqZS7XHRv6ycLpVedqc3mv7RNDmqoCSx4XVBu95ET6dR3+OtKEQ1DVNRiFg0NMMkaLFQm+IE0yQlGEANwLpvYWtqOjv62Bi3ppziq5cx8Zo+fPxVgB+2xFJwwkV+PvrzZqrWZDL5V92wpmh8/UUN782opsgb212N1+DlNyrp2sXKEUOcP37yozq88jl8/T0M7wVXTgFnK+Tgf7QUpn8L3bLgmhMhL33/6qv0wb/mwJYiOHEEnHfMgWjlgfdDEbzwKXgDcOEEGD8oVh4IwYufwQffQCgKJwyH606BrCamIP3wW7i5UV7qWwvAZYf/XNc6xyCEEPugTYP2DRs2sHHjRh588EGOPfZYnnnmGWbMmJEw4vaDDz5gx44dXHLJJQkBe73Bgwdz3XXX8fjjj/Paa69x4403tuYhCNEurXhta1JZyZZa0BJ7zWscNmodNlzBMAqgmpDmDVCW5QHTpHKLj6otPgC+mVHE07+cQHE0FlC/sMrk2sIcRmjFmEDI7cSw1AXbKqDHpoGMWlRKstO5cd4yPh3QnYLMNCzU8qdZb9Cv1MuazoNY06k3QUcapsVGRXoamh5FtyQG7pZGo+1DNisRa2xGmYiq4m2US6+YJp0qq7EbBt2DYfpU1ZAWCuMLwsd/+p4apxPS00jz1ZLuj+XEr/pgNzuXVhAdkMvG5V5KU5zxh0PV+255YN+C9kv/Bv/3VcP797+BeQ//+Hb74y/T4fevNLx/cS6seAoy3C2rLxiG8XfD+oLY+//MhTvPhj9dvP9tPZC2FMHo22K94wDPz4Z3fw/nHA2n/wk+W9Ww7oL18NqXsOIv4EpM1eK6fyfXPf1b+M/Ba7oQIpmJdLU3p02fiDp9+nRSUlL42c9+Rnp6OhMmTOCjjz7CqJs5AmDevHlArFd9b04//XQsFkt8XSEOZ8WrKwlXBJPKlUa/V42V5sR6HuszV8LWus/ysXT4uG/6dI4H7PX+b9QwFNNEt6gNAXsdU1Xi6S4ANsPkpM27uPLb9Vz99SaKbUOZ2/14irROjPphQ3zWgIjFgiMYxhqJ1FVk4giGSKkNoEWjsQGrlob+hqDVkjD41VQUalIcaIaBIxwmrW5gaz1PIIAa1UkNBBLKKwuCbF1SiSMcxtLEucrM3Ic8520liQE7wOdrYPH3P77t/njs/cT3O8rgjfktr2/6koaAvd5fPwR/qOV1HgzPz2oI2CH2Tc1j/4NvNyUG7PU2F8K7ixLLVm6Nna895Ug6lBCtzVCUpJdo0GZBeygUYtasWRx//PE4nbHeq6lTp1JcXMzXX38dX2/Lli24XC66d+++17ocDge9evVi165d+P3+va7XHlRUVBAKNfzh8/l8eL3e+PtwOEx5eXnCNoWFhc2+Lyoqig0elH3IPoilwliDkXiqCQCGiTUQAqNRmWmi6CZ6o0GaNW47NZ49eiHrBKzJc6X77VYUIGpvYh51RYlH/aYCulWLD4gNW+xUudPjefDbU3uR6o316KcEAlgNg1Sfn4yqGjKranD5A6iAFtXrBtcmBul7MpTYra3xh4Z4swDN0JtcppqxQbSZ/kDC+UuxGRw/saHXeq8/j+pamhIoLD94P3PDgOrk+160vKbl+6hq4jiCEXzlVe3r96Opdlb5qfihILk8vrw2cR9VTf/NqDlndOsdh+xD9tGG+xAdh2K20U9u1qxZ3Hvvvfzzn/9k9OjYzTEajXLKKadw5JFH8vjjjwMwduxYsrOz+eijj5qt74orrmDlypXMmjWL7OxsmT1GHLZev3wJ/kW7UQyTqM0SS3uJ6CimQdRqIWq3xWaUqRss2ntbGY5QhF1dM6jIcKEYBopZF/A26m3fle7midPGoTfqf5+4bRcXz1lGbXrdwNFGAbQO2ENhcqt8BG0WyrLSk9pq84dQ625B/hQHNWlu3LWBeFmc2fClqTfFia5pROu+EQhaLPjttoTVs6u9uEIhvClOUr0+LI3qi2oaFakeXIEAjkg0Xm6xqVT1zMZfGsIZiRDWVGptNjTT5Lq7ujHoyH3oeTVNGHoTrNvZUNYpHbb+Exy2vW21/y75K7z+ZcN7qwVWPw0DurasvsIKyL8usWf95JHwyX1736YtzF0JJzyUWHbfeXDPudDnN7C7InGZzQIb/w49cxvKIlHI/21ib3v3bNj6vMwiI0Qru/HcdUllf502uA1a0j61WU779OnTycjIIDc3l507G/7AjRs3jrlz51JVVUV6ejputxufz/ej9dXW1qKq6l4fGSvE4SKnn4d161w4vQEskVhueMrgTJSyWlydHaQNymTHwjL0ihB51bVkDc/k+zQPYRRSVdCDOjn5boKVYap3+lEVsNhVpp6cw5ipKo8uMdldC2f3U7igE6z8woKqGxiK0nhyGGzRKMO2FtC5yovXaefLPYN200zo7TY0FZuux3rOm+lLCNuspPgDaIZB2GrBFQhirwvQATyBIG6iBLPcFGe6+dzdmVOKisjyBeg2LI1wupPIdz4M1YbiN7AokNM3heOu6Ik9L4Vpr5ZQsLIaZyRKlyyTn13Qad8Cdoh9aPnoHrjlJVj0PYzoBX/+1cEN2AGevxo8jlgedtdMeOiClgfsAJ0zYdZ9cPcbsZSSk0bCXy47YM09YKYMh1dvgCc+gBo/XDwJ7v8FWDSYcz/c+F9YtCH2DdPgbvDnyxIDdoh9wJl9f2xMwNItMCY/dqwSsAvR6jriw5VaU5v0tO/atYuzzjqr2a9nbrnlFi688EKuvvpqli1bxvvvv7/XFJlgMMhxxx1HXl4e06dPB2SednH48hYHeff65VQVxHK2s/PdnPfsSJxpTaSw7Cc9YjDjjlXs/KIQ06LFe9pNoFdRGYN2NHwtu6FHZzZ3zQNFQVcUiERxhWJ562GbBV+aGxQFxTCwhSIN/fmN7hN+p4NaVwrWcJiuZSX4bSnx/el1+7c6VK55YzTuzIMcKAshhDigbjhvfVLZ394d1AYtaZ/apKd95syZmKbJvffei9udPLvB888/z4wZM7jwwgs5/vjjWbZsGR988AHXX399k/V9+OGH8dQaIQ53njwHl705lp3Lq1AtCt2Gp+/zg5J+Ks2q8vOnRvDeJbXs2OjHEtGxB2MBtyuQOGhx4I5C0mp8/O7yM9ie6eHGuUuxGQYhh51woykRTVUl6LARtFoxVBXVNLBEdVZ1ymJV1yzOWbmFcb3DRAevpSDUk9CK3Ngzl3QDFDjzrgESsAshRAdkSEd7s1o9aDcMg5kzZ5Kfn89ZZ53V5Do//PADL7zwAmvXruWss87inXfe4Y033mDUqFEcc0ziXMEbNmzgueeeIzs7m/POO68VjkCI9k+1qPQ8qvWeWaC6bTj9VaRVNgwMDDdx913esxNru2TRt7SK7GC4bgrKJgaLKgp2XSeoquiaxqbsdDbnpvGzwlJ+cVlnhp2WzauvrcFOFaddejxrPyzB0E2OOKMzvcbIsxqEEKIjkvSY5rV60L548WKKi4s588wz97rO8ccfzwsvvMD06dO5++67eeqpp7jhhhu46aabOP744xk1ahSaprF27Vo+/vhjUlNTeeqpp8jKymrFIxFC1PNVR3HVJE6hWONwsiMrg64VVWimyXd9u/DHMybSqcbPecs2ArHpvcJ2GyFVxa7roCj0qNhNRLNQmJZLSiiEqSh0r7FwZJrJZQ/1oHMPB5H66SCBbiPS6H1UdqserxBCCNHaWj1or885P/744/e6Tn5+Pj169GDOnDnccsst9OrVizfffJO33nqLefPmsXDhQgJ1cyz36dOHF198EY+niafcCSFahRk1UA0DQ4l9vakAqgHbszMpPS6fs/88nO5dPeQu9DL3b9vQU1wUOZxU2m28168bJvDPGTM4cdM3uCJBdqbmsaT7ULx2F+H8Tlz/3AAysiXlRQghDmWGPFypWa0etNdP5fhj/ve//yW8d7vdXHnllVx55ZVAbHrIO++8ky+++IIZM2Zw0UUXJazfpUsXli5demAaLYRoVvcjUtn5ze747dYEdNVEN02OuaE/jl6x2VemHJfKkF79WLO4Bleqhn1QGl12KGSnKJy8S8e5LkiJK4P3h/wMo/6JpLv8bJxVxNiLe7TJsQkhhGgdkh7TvDab8nF/WSwW/vSnP/H73/+ep59+GrvdzrnnntvWzRLisKT7o8mFikJuuY+sXomDzTv3ctK5lzP+flTv2P+NY3phvLaAdbl9GwL2Oqtn7pagXQghxGGtwwbtAFarlb/+9a9t3QwhDmtG1GDLV2U4m1hW63JQ88kOPNf8+JRdymXjUd5ZirK1iSeVatL7IoQQhzqZPaZ56o+vIoQQexesiRAOGISciTnnhqJQke3mu0+K96kexWFFm/d7hj91HNoe3QkjztmPBwUJIYToEAxFSXqJBh26p10I0fZSMu1k5bsp32SiGgYAhqbhdzswNJXCKv0n1Zf980H8clA3lr1bQNAbZdAJuQw6Ie9gNF0IIYToMCRoF0LstxP/cASz7lpJ2U4Nc498dMX5028znQZ6OPU+eQqeEEIcTmQgavMkPUYIsd+y+3m4eNqxuDolZ7Zn9pXpWIUQQvy4+mmDG79EAwnahRAHzOBTOyeVjbpIZn0RQggh9pekxwghDphxV/RGjxisn1WELUVj9EU96X20PK1UCCHEjzPl4UrNkqBdCHHAaBaVib/rx8Tf9WvrpgghhOhgZLaY5kl6jBBCCCGEEO2c9LQLIYQQQog2Jz3tzZOgXQghhBBCtDmZLaZ5kh4jhBBCCCFEOyc97UIIIYQQos0ZMntMsyRoF0IIIYQQbU6eiNo8SY8RQgghhBCinZOediGEEEII0eZkIGrzJGgXQgghhBBtTqZ8bJ4E7UKIVlVUHOGzL7yEQiYTjnHRL9/R1k0SQggh2j0J2oUQrWZ3YYR7HtpNIGAC8Ok8L7fekMvoI1PauGVCCCHamswe0zwZiCqEOGiigWjC+zmf1cQCdtNE03VME2Z8VN1GrRNCCNGe6ErySzSQnnYhxAFX8FUxix5aSc32WrKGpDPxT0eSNSgNX61Bt93lHLl2K65AmNIMD9tSBrZ1c4UQQoh2T3rahRAHVLAqzNzrvqFmey0A5WurmPvbxZiGydg+Csd8txFXIAxATqWX0Qs3tGVzhRBCtBOGoiS9RAPpaRdC7LeihSWUr6ggbVAa274pIxQ2UQHVMDEV8Bb4qdhYQ25hFTs0BW+KHdUwwIRQwGTDR4UMOLUTekBn9wc7CJeH6Dy1G64+nrY+NErXV1PwdSmp3V30npyHapG+DiGEOBhkysfmSdAuhNgv396/nI0vb8YEfB4HhkUFuwXdNLEFI2hRA1MDV66DAp9OYbc0UBQiVgtRmxWAmY9tZPOSSlI+3oJvYw0A6x9awZj/m0jeCV3b7NjWvLWNr/64Jv6+65gsTvvXOFRN/rIIIYRoXdJlJIRoMd/OWja+shmAiE2LBez1FIWIPdYvoJgmpm6yYUE5KAqmAlFrYp/B+nmllG/3x9+bEYMND688+AexF3pY55tnv08o27WknJ0LS9qoRUIIcWgzUJJeooH0tAshfrLSH2r54h9b2LWiGi0rFU+VD0U3cPiChB0WUvwRbCEdQ1MwFFB1k+lHTGdTdiYfHT2Q47YUkBXW4/VpkShOb4DqVCuKYUExTCy6SUWBn21DPsCMGigWlS7jchj3wHAcne0J7Sn4IcDM14sp2hmiz6AUzrqsE2mZ1maPoWpLDUv/tJqyNVXkjMhkzN3D8PRwx5eHfqgg7I0kbefdHdjPsyeEEKIpuuSwN0t62oUQP0kkpPPuzavY9k0lkZBB0OWgMicdBYjYLLhqIzgDUVQjNrWjagKqgmFRGFBcxmWfrOTZ40aiGEasQtPEXenFGo7EeuE1FcOiErWqGIqCHtQxoiZ6UGfn50XMuXwhpmHG2xMMGDz/8Ha+X1lLdUWU5Qtr+O8TO5s9BiNiMOfSBeycV0SgJMiOObv59NeJ9aZc/Tw5/vKE7RRNofv4nAN1KoUQQoh9Jj3tQoifZMuXpfBDJSmqQsDlwBEMY6gqEYuGriloEZ2opmAC7NFrErWpdC+r4cIF63BXeqlNc6NF9ViA35iiACZNfTNavdXHFzctxVphx5IdYUNGMbVeHXfAT/eKUopTM9i+CcqKwqTnWpm3I1b38T0UNMMg+vkWytdU4d/tJytcTWrYSwQH3vV+ylZXkjM8E3aWw1cbmGIrYF63Yyh2ZZMS8XP06dmkdXcdlPO6T8IRmLcarBY4bghoWtu1RQghDjAZiNq8Vg/aQ6EQM2bM4LPPPmPz5s14vV6cTic9evRg9OjRnHHGGfTq1Su+/syZM3nooYcAuPHGG7nkkkuS6tywYQMXX3wxAKeddhoPPvhgfNnpp5+O0+nknXfeOajHJcThYNfnhXx7zQI8dZ3kqZU+rBEDFQhrCpgmAWcsLUWL6th0M7kSE6as+gFDU3D7QnhTbE2s08R2jWz9ohTFSMPii1Dxz884pl8Kx29cjla33cIBQ6gwBjD+JZ2NlbFt+nsMZvzzP+Su2I4dk/OUalJMf12TFLzkEL0bzJkXobjsmJqGMwyn/rCEiKphMSLYu57XovN2QGwvgckPwNbi2PthPWHeQ5Cd2nZtEkKIA0iXHPZmtWp6TEFBARdffDGPP/44hmFw4YUXcs8993DttdeSn5/PjBkz+MUvfkFJSfJAL7vdzsyZM5usd8aMGdjt9iaXCSEOnEW3fQtGowIFopbYTVa3qAm3W11TE1YFsAd0ojYVo272Fc0w0RSFkGOPwN3c4/+NGFpDCk0wxULAZWHy9yviATvA+O/X8teFwXjADrDRq/KXnsMAsBCMB+yxwzBxUYH58QYiH2+ETDf6mEGYdf0aVkNHQSX6ceLA1Fb14NsNATvA6u3w5+lt1x4hhBCtqtV62oPBIDfddBMFBQU8+eSTTJ48OWmdUCjEm2++idLEQITjjjuO2bNns2bNGoYOHRovD4fDzJ49m8mTJzNr1qyDegxCHKp8lWEW/t9uirbU0n2Ih/Hnd8FiVVn0x9Vsnb0bxa4x5LJ8gqWhpG0NVSFkVYgqoOoGqmliomBoCiGrisUwUUzQorFAPuxI/P3WdJOi7lmk1PixBUO4fUHSa/w4IjpBq0ZhmovdqS7W5mbhikYJZKWwpnsennCU8duKoU+Qf3Q7Es0w+PXX3zDl+00ArJ+1m8sLq8mpqWV191zm9O3B2/mDyTm6ituXfAx64nFoRAGTmrNfw8xMQcnz4Cb2uWGTpxs7XTk418CI9dVkDkqLbfTVOnjmQ/AF4ZJJcPEkjEo/oce+ILp4O5YhOTjUKpTV28DthA0FUFWLObw3eqeumGt3o4QjKNlOVLuBEo3AicPh1jPBnjiQ1lyxFbDWfZAwUQijrNresEK5Fx77HyzZhJmZhlGrgN2Geu0klKnDG9Z79XN4fT6kOuHm02H8oJZcMkIIccDp0tHerFYL2j/44AO2bdvG5Zdf3mTADrHe9Msvv7zJZRMmTOCbb75h5syZCUH7l19+SXV1NaeffroE7UK0gKGbvHb7esq2x2ZF2bHay64NPvJ8Nez4tDC+3nd/XImN5K/nTAUiVq1uxpf6vnUTNWoS0RR0TQUTvOlO7KEoLl8wYfuIVcNUVWrT3dTihkofw7bEepS3ZXi49eTxhCyNblWp9tgLWNMli7ClIa979qABvPPiKxy7eSvnz1+Lra49A3eXk1dSw8tHDeGPE49jW5ad1z98LaEdUWyAgj3iRyv2YhYXU4ObrRl5rEvvFV9v9/lfcNYnJ+Au3A0/exAi0diCOSugNkjtK5vRv44F087581FI/qDD56uBbYANEwONKpT6rxW+XAvf74ZXbkjcxu7GxNFw3rHAwO4N326c8gf4Njb9pgKoaBikYny0CnXm9bHA/e8fw/X/aahzxlL45jEY2Se5jUII0crkCajNa7X0mHnz5gFw1llntWh7i8XCKaecwpw5cwiFGv4IzpgxgwEDBjBgwIAD0UwhDjvbV9XEA/Z42bIqdnxWmFCmmMTz1euZQFSN3WRNVUnIZlEgNnNMXWFKdYCw1UpE1cA0UXQTA9BVE2soHN+uKtXJ5t55GIrCzMG9EwN2gNqGdRsH7LE2qLw8bgybU7rFA/Z6x2zfjS0aC7DfGjSWSrsFs65xOho+MrAQQqvrglcAHQsbPYkPd4r4omz5YAf8Z25DwF6//6c+jAfsGiEsTQXs9eeGYN2/wyh75gG9MR9q/AlF5rbKxHVQMG3O2D+Xbo4H7A1LdSAKponxzy9jhc/PTqwiEoUXP2uyjUIIIdqXVgvat2zZgsvlomvXxD+Auq5TVVWV8AoGg03WceaZZ+L1evn8888BKC4u5ptvvuGMM8446O0/UCoqKhI+dPh8Prxeb/x9OBymvDxxmrnCwsJm3xcVFWE2yueVfcg+WrKPJE2NIa0LkqNKbOBpxKqBuvfbiGIChklWcYCu233krykhd5ePtNIQaaUhsoqCdN9STe7ucjKLYy9rOMKGAd1YOKYfXw7pkVSnOxjGEtWTd1bHrzgJKvswxkWJouAHAmh4ceDFRsMHAr9mY1GXQUS15PnevV5v04NlE8qaH0z7Y4oKixLe603sT6/fx48M3K1fHokkzzvfeNuOeu3KPmQfso+W76M90RUl6SUaKGYr/eTGjh1LdnY2H330UUL55s2bueCCCxLKGs8SUz97zGOPPcaUKVO49NJLcbvd/OMf/+DFF1/kxRdfZNasWRiGwZQpU2T2GCF+IkM3+efVKynf2fBhuefwVPK8Nexs1NtuKoCi4AhGqMp0Y/eH0Br1ZhuKgiPc0PNsArqq4K6OkFYZTig3tcQbcSDFQlGvVHRVpSwvC7Pug8Bn3fNY7EhBb/TB4KyNW/Fnu5lzRF+sUZ1I49520+TZFz9m7KYCNnTLIdpoSsSvenXhpTGx1Lrz1y3mrZnP03hOSQOVIJ3iZYtyBrHTnZt0viwuCz+fNQX37t1wzN3Q+APEP67G++pm9MU7AHCzs8nedhPQ8WBiAwwsjdNjAC6eBK/dmLjNQ+9jPvhBQ4HThrLyDyj9OsXeH3UbLN3SaB8aBh5AQZ15A8ppw+FvH8GNLzbUYbXA4j/BkX2T2iiEEK1t9G+Lk8qW/iOvDVrSPrVaTrvb7cbn8yWVd+3aleeeew6ATZs28cwzzzRbz+mnn86TTz5JYWEhH374IZMmTSI1NZWqqqqD0GohDn2qpnDJE4NZ+NYuijb76TbEzbG/7IrVqrLwD6vY9ulusGkM+VVfUlNUNk/bTlmhQdRmxeELYA3Fem8Lu+Xi9IeIoOBLSSGnrIZOpRU4/HukkDTRceLwRzEUjYqc9HjAjmly0qqtHKWazO/RmVqrlVFFpUwoKOIHXzqlaR6O3F3Gyi7ZfNc1G1NRQDeYPaQP2dV+uukhyjJTiWgqVQ4HGwd0oZ8e5MiQlzt3rmXPSeBVDMJOK1ElBdPloDQnDwKJfRqurin87F9H4+7qgq79YO6D8MxM8Abh0klw6WRc5/sJ/XEe0cU7CA8ehaZVxwaiepywviCW9jKsF3TqCmsLUSIRjKyeqHYdJRqFk0bAbWcln6T7z0LJScV8dwnkpqLcdmpDwA7wyX3wx/eSB6L+5rhYwA5ww1TwOGIDUdNSYgNRJWAXQogOodWC9r59+7Js2TJ27dqVkCLjdDoZO3YsANo+PCjk5JNP5plnnuGRRx5h586d3H777QetzUIcLjxZNk6+rndS+cQ/jGDiH0YklPX7ZR92TfmcGr9JIDUFtcKLappYozobe3clYItN37ijey7dC8o4/vO1mI0C5LrHJiWwmAauYJSSRvnrvTcUkVtUTXmOg4GV1YltKKxgUMHXAGQWVrA0Lyte64cj+7NsYn92XG9DUxv2+9v4v3Jgw7kw6KvERrgcpO9+BFJTAMi6fAG7vkzs9VEUGmaOAZg0JPZqRM1Mwfnn02iOwk+/+SqKAr/9Gcpvf9b0Ctmp8NTl8fr3eje9/GexlxBCtDOSDtO8VstpP/7444HYLDL7w+PxcNxxx/HNN9+Ql5cXD/iFEK0jUFBL6tpSFN0ARSHkdmAA1tpgPGCvt7NbNhUZKUQsDTdikz2eemeaOI0wXbaWk1I3s4w1FCGnKBaop1WG0KINaTiKYZLapWEWlYFF5QwtbMjptGvw7EmWhIA9ycBucPtZDe81Ff7yq3jADjDk1/lJm/kK/BR/W55ULoQQYv9FleSXaNBqPe1nnXUW06ZN47XXXmPw4MF7nfZxX1x22WX06NGDgQMHojYzCE4IceBFaiI4/BHydpTz32OH8UN6KmV2G8MrqunZxPo7e2bSqbgM1VRQdRNDg6hVwxGIklkWwIJJCCtaBCYW7KLaG8Vo1BVviZp03ukj6LSQc34vBlzVj8wj0njpz6+glFg49aozGeazUqyaBFKsTOqpkuvahzv945fAZZNjDyk6egB0z05Y7OqUQlqolu6+MtLDtdj1CKuyehGuCe+lQiGEEOLgabWg3eFw8Mwzz3DzzTdz2223MWrUKMaNG0dWVha1tbVs27aNTz/9FE3TyMtrftBBv3796NevXyu1XAjRWOrgdDxD0tHXV5NV7eOzHl0AWNgpm8yKGjxGQ6+4NRolmOUi5POj6okzl2zNSCWrLBR/xpFiVel961B2Xz8fgDJrClE1luShAGkujWP+MhqLyxqbBSXLwMwKk9vPTVdr8gwv+2RQt9irCWm9XRxTsQlHsGEg6biSTaQN+0XL9iWEEKJZUaRrvTmtFrQDdOvWjddee40ZM2bw2Wef8frrr+Pz+XA6nXTv3p0zzzyTM888k169erVms4QQP9HYtyax9u5ljN3so3bzDhZ3yUFFwW8adAqHCWoazkiE3GovClCd7aFrYRm7XE7Cqsri7nnU2qwE7FaG7iynZ38PvR8aQfYp3bDYFIqfXklurY4/04O/IoJnaDoD/zgKi6uFwXkL6CsLEwJ2AIuhw5IdcPrgVmuHEEIcLiISszer1aZ8FEIcema/X8bM/ytNKOuqhjDKEh/WNHhyFj+/pz/XvOtnxtxKfv/NKlx1DzoKO61c+s6xpPd27/N+I5EIL730EgCXX3451pb2tDfDKKyhpvsfYY+HNHlW3ox2ROcDvj8hhDjc5V9fmlS2+dmcNmhJ+yQJ4UKIFhs/JZ3svIaA2WJROP6K7jhTG77Es7s0xv8yNmPUbZPtnLmjIB6wA9gCEZa/mPg0z/ZA7ZyK/eZjE8qsF46QgF0IIQ6SiKIkvUSDVk2PEUIcWtweC3c+0ZtlX3vx+3SGj/GQ08nG0KNSWfdFGaYBg4/Lwp0Zm1UmP1vjZ5kRyncm1uPd7W+D1v8455OnYZk6CH3hNrSRXbCcPKCtmySEEIesJp7ZLBqRoF0IsV8cTo1jjk9PKHOlWznqrKZ7pPsel0f5ysqEsp4T2+8T76zH9cV6nDyASAghRNuSoF0I0aqGX9YX3+4AGz7YiaLAgJ93Z9jFyQ92EkIIcXjxSzpMsyRoF0K0Ks2qMvGBIxh/V+xJoprtx5+ELIQQ4tAXkJi9WRK0CyHahATrQgghxL6ToF0IIYQQQrS5sDxcqVkStAshhBBCiLYnMXuzZJ52IYQQQggh2jnpaRdCCCGEEG1PZo9plvS0CyGEEEII0c5J0C6EEEIIIUQ7J+kxQgghhBCi7Ul6TLMkaBdCCCGEEG1PYvZmSXqMEEIIIYQQ7Zz0tAshhBBCiHZAutqbI0G7EEIIIYRoexKzN0vSY4QQQgghhGjnpKddCHFICWyqZuefVhLa6iXj5G50uWUYqlX6J4QQot2TnvZmSdAuhDhkRCtDrBo/k0hpEIDqLwoJbK6h378ntHHLhBBC/DiJ2psj3U9CiA4tXBbEu76KSHmAwmdWEykNJCwveWUTuj/aRq0TQgghDgzpaRdCdFieGTY+v2MmLn8QdySEYoIdCKNhNu6TMM02a6MQQoh9JB3tzZKgXQjRIdk2aLi/tGLRo3jCoXi5AljRCdcF7dbBGWguaxu1UgghxL6TqL05ErQLIfZbTWGA1dN3E/RGGDAlj24jMw7avkzTZPcHO/F8bAPApienvqhAVIGIxQIZzoPWFiGEEKK1SNAuhNgvNYUB3rj8W4LVEQBWvb+LUx4YwsCTOh2U/a29bzlbntuALRrrkYmoWtI6UUWh1mpDURTcjuTlQggh2iHpaG+WDEQVQuyX1dN3E6iOYAAmsf98+9r2+PJoUG9yu72VNyfijbD1PxsxAWtQR4sahDQNv6Uh/cUEKt0pmJoCpkm4wI8RjGLqBgTDAOghHUNvIs89GAbD+MntEkIIcQAoTbxEnPS0CyH2y861NURsVhTAAFTTJOiNULqhhnkPr6F0Qw1p3VOYdNdgeozLZueiUhY8uprq7bVkD0pl0kPDyRmcvk/70gNRggYEPVZq0mzYA1FU06TY5cIejmKNGkQ0Dbc/jD0SjX2I2BBmsfO/WNUwPY0NWDupfKGNJJyVybDfDWLwVf2htBou/zt8vAyyPPDAL+B3px68kyaEEEL8RNLTLoRosYqdfnas8sY7Q1TAUFWyhqTz8S3LKN1QA0D1Tj+f3Lqcml1+5tz0LdXbawEoW1/D7BuXNt3r3YTakiCBFAumqqDqBqoCqAooCiG7lVqnDVcgjCMSRalrjxqN9ZxHDBubGYa9qJLxpV8Trgrz3SMrKVxQDNf+Cz76LjbLTFkNXP8fWLD+gJ4rIYQQP0a62psjPe1CHEaqyiMs+ayCim219O5tZ/jJuTg8DaklBSsqKVxdTXa+m8x8DxsWVGK1qwyemInNpfHJDybrv6kkv6icTm6VVav8AERVFd1iQdV1LLpO6c4QkaLYA45MQNdUwlGFF5/8AYdfp8bjImS34fb5oTBAxcYasgelxdsRKA+xZdYuFEWh98mdsS3Zhr5yN2WFDTdwzUgO9E0F7OHm5mRXqCSXruGt2KJBnGGTgnu+pnRZiKDrGCKahmoJ0N+3hYwX5hFaWII2JBfrqQNQ1Kb7OKIhnW2zd+MvCdJjcifS+3p+wk9ECCGE2DeKaR7YCYyXLl3KtddeG3+vqioul4ucnBwGDRrESSedxNFHH42iNPzxnTlzJg899NBe6+zWrRsffPABALt37+aMM85IWG632+natStTpkzh0ksvxeFwJLTlxhtv5JJLLjmARylEx7Nto5/nHthGJBz7lVd1nR5qgMueG056ZwcLnt/Mt6/GctF1TSXoSYmnd3uyrSw5dRBVs7Zz5rKN8TpDVgvVOZmE7PZ4mSUSwREM4fbGetPrf9NNwOtJwet0UpWWGl+/S0kpt757FCnZsToqNtUw45IFhOoGto4v20SPsqL4+htTu7AmswfWiIEtmph/bgtHyan0N3seerKNTLayzDGJtGCIhhtgrKU+u431XXMYVbae7jWxY7D+fAie/12cVFc0EGXmBfMpX1cdq0FTmPzUaPqc2q3ZNgghhEim3O1LKjP/6G6DlrRPB62n/aSTTmL8+PGYponf72f79u188cUXfPTRR4wZM4bHH38cjyexR+qCCy5g8ODBSXWlpKQklY0dO5apU6cCUFlZyaeffsoLL7zAqlWr+Pvf/35wDkqIDmzWO6XxgB3A0DRKAxrfvl3AMZd257s3dsSXhWy2hPGY3rIINZ/t5pSVmxPqtEeimHv0eEetViK6jqEoaGZiSJwSCBGs+1Bdr6RTNpq7obd/+b83xQP29KAvIWAHyK8pZGNaJ8JWGxbdQK3fhWmS6guhEiGiWNHqyk0aPjik4MdNkG32QU0G7ADuUJgMX4AVmf3oWrMSFZPI+2uJLNqO9ZieCW3Z8mFBPGAHMHWTb59cK0G7EEK0hCLpMM05aEH7wIEDOfXUxIFcN998M3/729944403uOeee/jb3/6WsHzEiBFMmTJln+rv0aNHQv3nn38+l156KYsXL2bt2rUMGTJk/w9CiEOAYZg8Ni/IlnVBnIpC2KJhKgoWw0DVNDZ/WsTKaTsIalYiDgtVDhtL87Kx6jrbUxwMqvYxrLKaUdsLmZHfi9W5maQHQ5z8w04GlVdh0XXCe+5TVdmd5qZ7lTdepqsquqbhqfVjKArlqR5QFKKGwi/eCrKiDGpCoKT0J3D+QEzgsiWLOWVbYt0qJk49gjWokFYVwqZHsVp9vH78EKaNHkN2jZ9j12/D1FUKszxsz0mjc6WPVb1zqXQ7sUej3PfePE5dsbWuxuQ/EsNLt5Nm1qCrKt+ndWZjeidcF89nRGox+fdNwPd5GcHpG7CrFgaW23CFokQ0jYKMNCp2msw84gOOfHw0XU/phndZGVvv+Bb/uirSjutM37+MwdYpuSNCCCGEaE6rDkTVNI2bb76ZESNGsGjRIlasWHHA6rZYLIwZMwaAnTt3HrB6hejonp4f4p6Pg2yz2ghaLRiqiqkoRDSNkMVCtKgWTAVNU3EYBp38QU7dWsAmt4t8r59xZZW4dIOPu3flqx6dqXLY2Zaeyr9HDKI4xUHAmdhzjmmyI81DYUZDjrqhKERtFlAVVNMkw+sjqzo2SHVFTiozymzswEaV3Ual20nQYSVks/LWkaPw2mwJ1QdVK+GojdzSAPaIgWKoREOpBKMZlKW62NAth1cmH0muz8/Ho/uxtlcec0f2pSTdQ8Riwedw8JfTxwNGXbi+Z4agicesxYKB3YgytHInvbxl1Ghu5tf2pfCCD/A/twSjoAZ9R4DM2hD2qI47FGZAUSlp/iDBkiCLfr2A8m9KWH3CLKrm7ia820/pm1tYd/ZnB/pHLIQQ4jDQJrPHnHnmmQAsWLAgodzv91NVVZX0CgQC+1Tvjh2xr/fT09MPaHuF6Mje+C7WD+61WpK+egzb7ZhAxGpNKFeBIyqqGFYZS/0wgbVpielsTsNgdc+uYBgodbk0JrEAvSDVTVmqh/L0VAwFdIuatG+PP0CJzcJ3eenJjVZj61Y7nfzq/AupqPtgUOZ0sSItn1RvJGmTKcu3xv8dsWhsz00nv7Aci548H3yZ20OethErwbrjrR+8apBBJTYS6+/hK4v9wzRRow0fIsIkfqBQgNzqupxME7Y9s5ZoRShhnZqvSwhs9SKEEGIPMnlMs9okaO/Xrx8A27dvTyh/+OGHmTJlStJrzzQagHA4HA/qt27dyj/+8Q/mz59Ply5dOPLII1vlOFqioqKCUKjhj7jP58PrbfgDHg6HKS8vT9imsLCw2fdFRUU0Hk8s+5B9NN5HmiN21wtoyb/uqmnU3ReTx6MHNI2gFnuaqALY9YYk9+HVXq7YUUh+OELUZsURDJJVXo7b5wPTwKIb6KpKaXYGG/v0oKLRwNN6Fl1nYEFhPODfm8/69eOWk85hWr9xLMgdRoXDTVRLvpP7HIkBtCsY4fbpC7jq42VJ61r1KJlGCf35hsEsYAgL6Kkt5Zu+PdiSlZO0fvypq4qC0WjXTZ03Q2k4z5ZMe9JyNAXDZnb460r2IfuQfRwa+xAdx0GbPaa5GVt27tzJz3/+c8aMGcM//vGP+OwxV111FSNGjEhaPy8vj169egFNzx5T78gjj+Tee++lR48e+9wWIQ51H6+LcPqLPiy6wblllTgbDRy1W3QGrt1OxGohlNKQ5uKzWnh2cD69fH7O3bEbgK8z05mXl41d17l6RyGWxrcO0yS7vBKLYeBzOSlI9bA2N5u+Xh/uqI4W1elaWpYwTaNixD4wfNy3G/OO6I1RP6WiaULUrOu2N0n1B7n344WkBcMouoEjFMUSNelUVBsfhGoCD148kS+H9wIgp7qWR9/4DLc/QkpVlGtvmUpRZsMMBK5QgKIHr8UdbfgWb2HOGNanDQDghB3LSQ/HlhnAV10GU5KSjsWIMLVsKXqNE4AgNvy44nVEFZUtWVlENQ1rmpWTvjyF9ad9iu+7svg6na4aQP8Xjv1JP0MhhDgcKPcmz/5lPiJjgOq1yTzttbWxadRcLldCed++fRk7duw+1TFp0iR+8YtfoCgKNpuN7t27k5WVdcDbKkRHd+pgK4tu8PDKt2GsESupO2qpKo/Se1AK116YwdYvM/jq3d1Ubgvhc9godDnZmp7KpCovaUemUTS0D312VnBmWS09fQYFbldiwA6gKHzfOY+A1YLbH2BTmotJm7eyIyOdsGmSFg4TslpRdB1HJPYU0/oO6+P9VVx5qsrHK0IU7g6RYlepzXViaAojdm4n95MfSAuG4/tBUYhaFYo6uXD7wigmBNKD+NIg0+unZ3EVv/tgCa5oFKc/SkpI529/m8WHx/SnJD2FMet30S9QyGpzLFnW3RhOg22uHhS4usYPZ5cth5TwDiJ9ctgydAhKAIaEKxg03EL6HXcSXFpCaMb3uLp6yBzcBd9nBRiayq5Sk5TCIBlHZjL0ruE4O6VwxLxTKPzXhvhA1LxL8lvnBy+EEB2NpMM0q02C9k2bNgHEe89bIjc3d58DfCEOd2N7Whjbs/7XPTFVZciJeQw5MY8/PLCTqi0hnMBgX6yX+ffH2RkyLANoSBkJBXT+eFkVoUBDWosBFKanElVVyjxuBgVq6OXUydm5O76OCdSmOBOmgQQ4emIGxw21cP5QC5D4Qd6I5PPGx1uoH9ZiKg1TOEatKlUZDspdDu6+aAoBeywvv8KTwuxR/fj1rJWxXnsgozbEJZ+urjv6GobyPfMdoylXMynJcmI0Th0yTaqDadgpJ+fVX9Nl/KCk8+k4IxPHGQPj71PPj/27qYkeLak2ut92RBNLhBBCiH3XJjnt06dPB2D8+PFtsXshRBMuuyKXnJxYYK9pcNKp6QwZlvy1pN2pcd5NPbDZY10iBlCQ6iHa6Imhet8MTnpwKM6MWJ65CfjcLmpdKajpDXneXUekM+7XvffaJtWqMvmxUTjq6tGcFrRsRzyTXDFMlvbuEg/Y6300LjZuxkYUFwHqZ4hxECCfrVSq6fQKl2AxddKrwqh1aTuKYZJVFmCAsYHUe6ZAEwG7EEKIg0QGojarVXvadV3n2WefZcWKFYwfP77J/HUhRNvo3sPOY3/pScHOMGnpGmlpe789DD06nf6vp7J7s5//zvRSsSVxMKnTqdF9dCZXzJxA+Q8+VLtGeXGYzG5Ocro6qCrwYxqQ0ePHcxW7jc/loi9PonKzF0+3FD66aAEV1RE03SC7OkDEpiVvpEA3owwbsZljdFTAIJ1q1tOfkGohxQiTHvDT78bBdL11KDW7AqhVIVINL7ahp0FOWnK9QgghDiKJ0ptz0IL2DRs28PHHHwMkPBG1sLCQcePG8eijjyZts2LFCsLhPR/TEnPKKaegyJOyhDioVFWhR88mZjxpgs2u0muIm3MsVpb+pYzGD0Y9dVIszUWzquQOiKXjZPdqSH1J7/bTBhZpNo3swekADPplbxY+sAJTValJdXLM9wXMOGoAQVtDb/uxG7ajKgaYEFatGIYGaFSQjYKBI2pgYMGSYaf7zUOxZTtxZDvrtu78k9omhBBCtIaDFrTPnj2b2bNno6oqTqeTvLw8jjzySE466SSOOeaYJrd566239lrfiSeeiMXSJin4QohmDB9g5w83ZjFjXi2RqMmJ41M4bszBG+0/6MLeKDb4+tElUK6S7Qtw37T5zBzVnyq3k2E7ijl+3VaqHU5cgQju3xyJWqET3uUn8/QeEIlS/clO7D3ddL1rJLZc54/vVAghxMEnfbPNOuBTPgohxMEWiUR49e7XcbztTig3FAg4LCgmDP2hBKtuMOuPp3LvXT3bqKVCCCH2lfJA8sM0zYekY6VemwxEFUKI/aXnR4gOakinM4FwXX57lzJvLGAf3peHtM6U+aVvQgghRMcm+SZCiI5JgdA5tZw86gRCRWGc3VKYtaSW+9fb6Vtcya7MVDZ0ywYTfBHIbuv2CiGEaJ6kxzRLgnYhRIeWeUQG1lGxQahnDMzghuejfJaXEV8+rotCrzT5SyCEEO2eTDjSLEmPEUIcMtLsCnPO15jcQyHPBecPVPjfz5uYElIIIYToYKSnXQhxSDmqs8q8X0p/hBBCiEOLBO1CCCGEEKLtSXZMs6Q7SgghhBBCiHZOetqFEEIIIUQ7IF3tzZGgXQghhBBCtD2J2Zsl6TFCCCGEEEK0cxK0CyGEEEII0c5JeowQQgghhGh7kh7TLOlpF0IIIYQQop2ToF0IIYQQQoh2TtJjhBBCCCFE21MkP6Y50tMuhBBCCCFEOyc97UIIIYQQou1JR3uzpKddCCGEEEKIdk6CdiGEEEIIIdo5SY8RQgghhBBtT9JjmiVBuxBCCCGEaAckam+OpMcIIYQQQgjRzklPuxBCCCGEaHvS0d4s6WkXQgghhBCinZOgXQghhBBCiHZO0mOEEEIIIUTbk/SYZklPuxBCCCGEEO2cBO1CCCGEEEK0c5IeI4QQQggh2p6kxzRLetqFEEIIIUSH8uCDD+J2u9u6Ga1KgnYhhBBCCCHaOUmPEUIIIYQQbU+R/JjmSE+7EEIIIYRoe0oTrxZavXo1J510Ei6Xi7S0NM4991x27NgRX37FFVcwYcKE+PuysjJUVeWoo46Kl/l8PqxWK++++27LG3IASU97KzJNE6/X29bNEKLDi0QiBAIBAGpqarBarW3cIiGE6Lg8Hg/KIdTLvXPnTiZOnEjfvn15/fXXCQaD3HPPPUyaNIlVq1bh8XiYOHEib7zxBsFgEIfDwfz587Hb7Sxfvhyv14vH42HRokVEo1EmTpzY1ocESNDeqrxeL2lpaW3dDCEOKTfddFNbN0EIITq06upqUlNT27oZmL8/MGHp008/TSQSYc6cOWRmZgIwcuRIBg8ezMsvv8z111/PxIkTCYVCfPPNN0yaNIn58+fz85//nDlz5rBw4UJOPvlk5s+fT//+/cnLyzsg7dpfErS3Io/HQ3V19T6v7/P5mDp1Kh999NFhN0J6T3IuYuQ8NJBz0UDORQM5Fw3kXMTIeWiwt3Ph8XjasFUH3ldffcXxxx8fD9gBBg4cyPDhw1mwYAHXX389vXv3plu3bsyfPz8etF977bUEAgG+/PLLeNDeXnrZQYL2VqUoyk/6JKuqKpqmkZqaetjfaORcxMh5aCDnooGciwZyLhrIuYiR89DgcDkXlZWVjBgxIqk8Ly+PioqK+Pv6YL2mpoaVK1cyceJEamtrmTZtGqFQiCVLlnDVVVe1YsubJwNRhRBCCCHEISMzM5OSkpKk8uLi4oTe94kTJ/L111/zxRdfkJ2dzcCBA5k4cSLffvstn3/+OaFQKGGwaluToF0IIYQQQhwyjj32WD777DMqKyvjZd9//z2rVq3i2GOPjZfV96w/9dRT8TSYESNG4HQ6eeyxx+jevTu9evVq7ebvlaTHtGM2m42rrroKm83W1k1pc3IuYuQ8NJBz0UDORQM5Fw3kXMTIeWhwqJ0LXdeZNm1aUvmNN97ISy+9xIknnsg999xDMBjk3nvvpUePHlx22WXx9QYOHEhubi5ffvklf/vb3wDQNI3x48fzySefcNFFF7XWoewTxTRNs60bIYQQQgghxL568MEHeeihh5pc9tprr3HEEUfw+9//noULF6JpGieccAJPPfUUPXv2TFj3vPPOY9q0aaxYsYLhw4cD8Pjjj3PnnXfyr3/9i6uvvvqgH8u+kqBdCCGEEEKIdk5y2oUQQgghhGjnJGgXQgghhBCinZOBqO3I4sWLmTlzJmvWrGHXrl2cd9553HHHHT+63e7duznjjDOSyocOHcrLL798EFp68LX0XEDs4RFPPfUUX3zxBdFolHHjxnH77beTnZ19kFt98MyfP5/nn3+e7du306lTJy677LImf+aNdeTrYtu2bTzxxBOsWrUKl8vFqaeeym9/+1usVmuz25mmySuvvMK7775LVVUV/fv355ZbbmHYsGGt1PIDr6Xn4vTTT6ewsDCpfOHChdjt9oPV3INq586dvPbaa6xZs4YtW7bQs2dP3nnnnR/d7lC7Llp6Hg7Fa2Lu3Ll8/PHHbNiwgZqaGnr06MH555/PGWecgaIoe93uULsmWnoeDsVr4lAmQXs78vXXX7Np0yaOPPJIampqfvL21113HaNHj46/T0lJOZDNa1X7cy7uuusufvjhB+666y5sNhv/+Mc/uOGGG3j11VexWDreJb9ixQpuu+02zjzzTG699Va+/fZb/vCHP5CSksKUKVN+dPuOdl3U1NRw7bXX0qNHD5588klKSkp4+umnCQaDP/rB7ZVXXuFf//oXv/vd7+jXrx/vvvsuv/vd73jjjTfo1q1bKx3BgbM/5wLgZz/7GRdffHFCWUeeNWLLli0sXLiQIUOGYBgGhmHs03aH2nXR0vMAh9418cYbb9C5c2duuukmMjIy+Oabb3j00UcpLi5udgDhoXZNtPQ8wKF3TRzSTNFu6Loe//dpp51mPvbYY/u03a5du8xRo0aZn3766cFqWqtr6blYuXKlOWrUKPPrr7+Ol23dutUcPXq0OWfOnAPeztZw3XXXmZdffnlC2d13322ee+65zW7XUa+L//73v+axxx5rVlVVxcvee+89c8yYMWZJScletwsGg+bEiRPNv//97/GycDhsnnbaaeaf/vSng9rmg6Wl58I0f9rvTUfR+L7wwAMPmOedd96PbnMoXhctOQ+meWheE5WVlUlljzzyiDlx4sSE89TYoXhNtOQ8mOaheU0cyiSnvR1RVflx1GvpuVi0aBEej4exY8fGy3r16kX//v1ZuHDhgWpeqwmHwyxdujSpR/3EE09k69at7N69u41advAsWrSIMWPGkJaWFi874YQTMAyDxYsX73W7VatWUVtbm3CurFYrkydP7pA/e2j5uThUteS+cCheF/K3okF6enpS2YABA6itrSUQCDS5zaF4TbTkPIiOR37zDyGPPfYYY8aM4YQTTuCRRx6hurq6rZvU6rZt20bPnj2Tcvh69+7Ntm3b2qZR+6GgoIBoNJr0RLbevXsD7NMxdbTrYtu2bUnH6/F4yM7ObvZ465c1da6KiooIBoMHtqGtoKXnot6sWbM4+uijmTBhAjfccAObN28+OA1txw7F62J/HA7XxIoVK8jNzcXlcjW5/HC5Jn7sPNQ7HK6JQ0XHS/AVSWw2G+eeey7jxo3D4/GwZs0a/vvf/7Ju3boOm8fdUjU1NXg8nqRyj8fTonECba2+zXseU2pqasLypnTU66KlP8OamhpsNlvS4CmPx4Npmni9XhwOxwFv78G0P9fzxIkTGTp0KJ06dWLXrl3897//5YorruiwObstdSheFy11OFwTK1asYM6cOdx00017XedwuCb25TzA4XFNHEra51/tQ4TP56OsrOxH1+vateuPzgTRnOzsbO688874+1GjRtG3b19uuukmPv/8c0444YQW132gtNa56Ah+yrnYHx3huhAHz2233Rb/98iRIxk3bhznnHMOr7/+esJ1IQ4fh/o1UVxczF133cXo0aO54IIL2ro5beannIdD/Zo41EjQfhDNnTuXRx555EfXmzZtWtLXdPtr/PjxOJ1O1q9f3y6Cs9Y6F6mpqRQXFyeVe73eeO90W/sp56K+zT6fL2FZfS/rTz2m9nZdNCU1NTXpeOHHf4apqamEw2FCoVBCD5rX60VRlCZ7rNu7lp6LpmRnZzNixAjWr19/oJrXIRyK18WBcihdE16vlxtuuIG0tDSeeOKJZvP+D+Vr4qech6YcStfEoUiC9oPorLPO4qyzzmrrZrQLrXUuevXqxZIlSzBNMyGvfdu2beTn5x/0/e+Ln3IuwuEwFouFbdu2cfTRR8fL95aTeSjo1atXUr52/bcTzR1v/bLt27fTv3//ePm2bdvo1KlTh/y6u6XnQjQ4FK8LkSgYDHLTTTfh8/l46aWXcLvdza5/qF4TP/U8iI5HBqIeor766isCgQCDBw9u66a0qmOOOYaamhqWLFkSL9u+fTvff/8948ePb8OWtYzNZmP06NF89tlnCeWffvopvXv3pkuXLj+pvo5wXRxzzDEsWbIEr9cbL5s7dy6qqjJu3Li9bnfEEUfgcrmYO3duvCwajfL55593yJ89tPxcNKW0tJQVK1a065/9wXAoXhcHyqFwTUSjUe666y62bdvGs88+S25u7o9ucyheEy05D005FK6JQ5n0tLcjhYWFrF27Foh9Yt61a1f8ptJ4aqqxY8cydepU7r//fgCefvppVFVl6NCheDwe1q5dy8svv8zgwYM57rjjWv04DoSWnosjjjiCo48+mocffpibb745/nClfv36MXny5NY/kAPgyiuv5JprruGxxx5jypQpfPfdd8yaNYs//elPCesdKtfFOeecw9tvv82tt97Kr3/9a0pKSvjrX//K2WefTU5OTny93/zmNxQWFvLBBx8AYLfbufzyy3nhhRfIyMggPz+fd999l+rq6qQHh3QULT0Xs2bNYsGCBYwfP56cnBwKCgp4+eWX0TStw54LiN0LFixYAMTuEbW1tfH7wqhRo8jIyDgsrouWnIdD9Zp4/PHH+eqrr7jpppuora1l9erV8WUDBgzAZrMdFtdES87DoXpNHMokaG9Hli5dykMPPRR/v2jRIhYtWhRfVk/X9YQn4PXu3Ztp06bxv//9j2AwSG5uLmeccQbXXHNNu50h5Me09FwA/OlPf+Kpp57i0UcfRdd1xo4dy+23395hz8WIESN44okneP7555k+fTqdOnXi3nvvTZq7/VC5LlJTU3n++ed58sknufXWW3G5XJx11ln89re/TVhP13V0XU8o+9WvfoVpmrz++utUVlbSv39/nn322Q47C0JLz0XXrl0pLS3lL3/5C16vF4/Hw1FHHcU111yz3wOc21JFRUXS4Lj69//85z8ZPXr0YXFdtOQ8HKrXRP3zCp555pmkZTNmzKBLly6HxTXRkvNwqF4ThzLFNE2zrRshhBBCCCGE2DvJaRdCCCGEEKKdk6BdCCGEEEKIdk6CdiGEEEIIIdo5CdqFEEIIIYRo5yRoF0IIIYQQop2ToF0IIYQQQoh2ToJ2IYQQQggh2jkJ2oUQQggh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
"/>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<h2 id="Insights-from-SHAP-Analysis-for-XGBoost-Model">Insights from SHAP Analysis for XGBoost Model<a class="anchor-link" href="#Insights-from-SHAP-Analysis-for-XGBoost-Model">¶</a></h2><p>The SHAP (SHapley Additive exPlanations) plots provide valuable insights into the feature importance and how each feature influences the XGBoost model's predictions for earnings manipulation.</p>
<h3 id="General-Interpretation:">General Interpretation:<a class="anchor-link" href="#General-Interpretation:">¶</a></h3><ul>
<li><strong>SHAP Value:</strong> The SHAP value for a feature represents the average change in the model's output prediction when that feature is included in the model, compared to when it's not. Positive SHAP values indicate features that push the prediction towards the positive class (Manipulator='Yes'), while negative SHAP values push it towards the negative class (Manipulator='No').</li>
</ul>
<h3 id="Bar-Plot-(Global-Feature-Importance):">Bar Plot (Global Feature Importance):<a class="anchor-link" href="#Bar-Plot-(Global-Feature-Importance):">¶</a></h3><ul>
<li>This plot quantifies the overall importance of each feature. The features are ordered from most impactful to least impactful based on the average absolute SHAP value across all samples. A longer bar signifies a greater overall influence on the model's predictions.</li>
<li>To understand which features are globally most important, observe the top features listed in the bar plot.</li>
</ul>
<h3 id="Beeswarm-Plot-(Individual-Feature-Impact-and-Direction):">Beeswarm Plot (Individual Feature Impact and Direction):<a class="anchor-link" href="#Beeswarm-Plot-(Individual-Feature-Impact-and-Direction):">¶</a></h3><ul>
<li>This plot shows the distribution of SHAP values for each feature across all data points. Each dot represents a single instance's SHAP value for that feature.</li>
<li><strong>Horizontal Position:</strong> The position of each dot on the x-axis indicates the SHAP value for that instance. Dots on the right (positive SHAP values) contribute to predicting 'Manipulator=Yes', while dots on the left (negative SHAP values) contribute to predicting 'Manipulator=No'.</li>
<li><strong>Color:</strong> The color of the dot typically represents the feature value for that instance (e.g., red for high feature values, blue for low feature values). This helps in understanding the <em>direction</em> of the relationship:<ul>
<li>If red dots are clustered on the right (positive SHAP values) and blue dots on the left (negative SHAP values), it suggests that high feature values increase the likelihood of manipulation, and low feature values decrease it.</li>
<li>Conversely, if blue dots are on the right and red dots on the left, it implies that low feature values increase the likelihood of manipulation, and high feature values decrease it.</li>
</ul>
</li>
<li><strong>Vertical Spread:</strong> The vertical spread of dots for a feature indicates how many data points have SHAP values for that feature, and if there are many overlapping points, the density is higher.</li>
</ul>
<h3 id="Actionable-Insights-(Based-on-typical-outcomes):">Actionable Insights (Based on typical outcomes):<a class="anchor-link" href="#Actionable-Insights-(Based-on-typical-outcomes):">¶</a></h3><p>By carefully examining both plots, you can derive specific insights, for example:</p>
<ol>
<li><strong>Top Influential Features:</strong> Identify the top 2-3 features that consistently have high average SHAP values from the bar plot. These are your model's most critical predictors.</li>
<li><strong>Direction of Impact:</strong> For these top features, observe the beeswarm plot to understand whether high or low values of these features are associated with a higher likelihood of earnings manipulation.</li>
<li><strong>Interaction Effects/Complex Relationships:</strong> The spread and clustering in the beeswarm plot can sometimes hint at more complex interactions or non-linear relationships that the model has learned.</li>
</ol>
</div>
</div>
</div>
</div>
</main>
</body>
</html>
