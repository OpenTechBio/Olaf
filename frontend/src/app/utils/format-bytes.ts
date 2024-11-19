/**
 * Receives a number of bytes and formats into MBs, GBs, etc. as a string.
 *
 * Reference: https://stackoverflow.com/questions/15900485/correct-way-to-convert-size-in-bytes-to-kb-mb-gb-in-javascript
 *
 * @param bytes The number of bytes.
 * @param decimals The number of decimals to truncate to.
 * @returns A formatted string containing the size in MBs, GBs, etc.
 */
export function formatBytes(bytes: number, decimals = 0): string {
  if (!+bytes) return "0 bytes";

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = [
    "bytes",
    "KB",
    "MB",
    "GB",
    "TB",
    "PB",
    "EB",
    "ZB",
    "YB",
  ];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}
