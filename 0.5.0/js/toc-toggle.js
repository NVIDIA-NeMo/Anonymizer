// Toggle TOC sidebar sections for better navigation on long pages
document.addEventListener("DOMContentLoaded", function () {
  // Auto-expand the active TOC item on page load
  const activeTocItems = document.querySelectorAll(".md-nav__link--active");
  activeTocItems.forEach(function (item) {
    let parent = item.parentElement;
    while (parent) {
      if (parent.classList && parent.classList.contains("md-nav__item--nested")) {
        const toggle = parent.querySelector("input.md-nav__toggle");
        if (toggle) {
          toggle.checked = true;
        }
      }
      parent = parent.parentElement;
    }
  });
});
