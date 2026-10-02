// 拡張機能のアイコンを押したら、管理ページ（launcher.html）を開く。すでに開いていればそのタブを前に出す。
// 既存のタブの探索に失敗しても、必ず新しいタブを開く。
const LAUNCHER_URL = chrome.runtime.getURL('launcher.html');

async function findLauncherTab() {
  try {
    const contexts = await chrome.runtime.getContexts({ contextTypes: ['TAB'], documentUrls: [LAUNCHER_URL] });
    return contexts[0] ?? null;
  } catch (error) {
    console.error(`既存のタブを探せない: ${error}`);
    return null;
  }
}

chrome.action.onClicked.addListener(async () => {
  const existing = await findLauncherTab();
  if (existing) {
    try {
      await chrome.tabs.update(existing.tabId, { active: true });
      await chrome.windows.update(existing.windowId, { focused: true });
      return;
    } catch (error) {
      console.error(`既存のタブを前に出せない: ${error}`);
    }
  }
  await chrome.tabs.create({ url: LAUNCHER_URL });
});
