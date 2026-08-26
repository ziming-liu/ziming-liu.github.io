// Progressive enhancement only: the pages are fully usable without this file.

document.addEventListener('DOMContentLoaded', () => {
  const year = document.getElementById('year');
  if (year) {
    year.textContent = String(new Date().getFullYear());
  }

  if (!navigator.clipboard) {
    return;
  }

  document.querySelectorAll('.bibtex pre').forEach((pre) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'copy-btn';
    button.textContent = 'Copy';

    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(pre.textContent.trim());
        button.textContent = 'Copied';
      } catch {
        button.textContent = 'Failed';
      }
      setTimeout(() => {
        button.textContent = 'Copy';
      }, 1500);
    });

    pre.parentElement.insertBefore(button, pre);
  });
});
