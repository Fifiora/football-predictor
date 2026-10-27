export const STALE_AFTER_HOURS = 48;

export function pct(p) {
  return `${Math.round(p * 100)}%`;
}

export function slug(text) {
  return String(text)
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

export function isStale(updatedAt, now = new Date(), hours = STALE_AFTER_HOURS) {
  const updated = new Date(updatedAt).getTime();
  if (Number.isNaN(updated)) return true;
  return new Date(now).getTime() - updated > hours * 3600 * 1000;
}

export function topDrivers(drivers, max = 3) {
  return [...(drivers ?? [])].sort((a, b) => b.strength - a.strength).slice(0, max);
}

export function formatUpdated(updatedAt) {
  const d = new Date(updatedAt);
  if (Number.isNaN(d.getTime())) return "unknown";
  return d.toISOString().replace("T", " ").slice(0, 16) + " UTC";
}
