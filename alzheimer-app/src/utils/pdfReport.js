import { jsPDF } from "jspdf";
import autoTable from "jspdf-autotable";
import { CLASSES, CLASS_LABELS } from "./constants";

const MARGIN = 14;
const PAGE_WIDTH = 210;
const CONTENT_WIDTH = PAGE_WIDTH - MARGIN * 2;

const COLORS = {
  primary: [15, 61, 92],
  headerBg: [232, 241, 248],
  altRow: [248, 250, 252],
  predicted: [237, 247, 241],
  border: [212, 222, 232],
  muted: [90, 107, 125],
};

const loadImage = (src) =>
  new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });

const compressImage = (img, maxWidth = 320) => {
  const canvas = document.createElement("canvas");
  const ratio = Math.min(1, maxWidth / img.width);
  canvas.width = img.width * ratio;
  canvas.height = img.height * ratio;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.78);
};

const addFooter = (doc) => {
  const pageCount = doc.getNumberOfPages();
  for (let i = 1; i <= pageCount; i += 1) {
    doc.setPage(i);
    const pageHeight = doc.internal.pageSize.getHeight();
    doc.setDrawColor(...COLORS.border);
    doc.setLineWidth(0.2);
    doc.line(MARGIN, pageHeight - 16, PAGE_WIDTH - MARGIN, pageHeight - 16);
    doc.setFontSize(8);
    doc.setTextColor(...COLORS.muted);
    doc.text(
      "This report is a diagnostic aid. It does not replace medical judgment.",
      PAGE_WIDTH / 2,
      pageHeight - 11,
      { align: "center" }
    );
    doc.text(`Page ${i} of ${pageCount}`, PAGE_WIDTH - MARGIN, pageHeight - 11, {
      align: "right",
    });
  }
};

const addSectionTitle = (doc, title, y) => {
  doc.setFont("helvetica", "bold");
  doc.setFontSize(11);
  doc.setTextColor(...COLORS.primary);
  doc.text(title, MARGIN, y);
  doc.setDrawColor(...COLORS.primary);
  doc.setLineWidth(0.4);
  doc.line(MARGIN, y + 2, MARGIN + 55, y + 2);
  return y + 8;
};

const defaultTableStyles = {
  theme: "plain",
  styles: {
    font: "helvetica",
    fontSize: 9,
    cellPadding: 3,
    lineColor: COLORS.border,
    lineWidth: 0.1,
    textColor: [30, 45, 58],
  },
  headStyles: {
    fillColor: COLORS.headerBg,
    textColor: COLORS.primary,
    fontStyle: "bold",
    fontSize: 8,
  },
  alternateRowStyles: {
    fillColor: COLORS.altRow,
  },
  margin: { left: MARGIN, right: MARGIN },
};

const getFinalY = (doc, fallback) =>
  (doc.lastAutoTable?.finalY ?? fallback) + 8;

export const generatePdfReport = async ({
  analysisDateTime,
  originalPreview,
  gradcamBase64,
  prediction,
}) => {
  const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  const date = new Date(analysisDateTime);
  const dateStr = date.toLocaleString("en-US", {
    dateStyle: "long",
    timeStyle: "short",
  });

  const { predicted_class, confidence, probabilities } = prediction;

  // --- Header ---
  doc.setFillColor(...COLORS.primary);
  doc.rect(0, 0, PAGE_WIDTH, 28, "F");
  doc.setFont("helvetica", "bold");
  doc.setFontSize(16);
  doc.setTextColor(255, 255, 255);
  doc.text("Alzheimer MRI Analysis Report", PAGE_WIDTH / 2, 14, { align: "center" });
  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.text("DenseNet121 + CBAM  |  Clinical decision support", PAGE_WIDTH / 2, 21, {
    align: "center",
  });

  let y = 36;

  y = addSectionTitle(doc, "Report information", y);

  autoTable(doc, {
    ...defaultTableStyles,
    startY: y,
    tableWidth: CONTENT_WIDTH,
    columnStyles: {
      0: { cellWidth: 52, fontStyle: "bold", textColor: COLORS.primary },
      1: { cellWidth: CONTENT_WIDTH - 52 },
    },
    body: [
      ["Analysis date and time", dateStr],
      ["Model", "DenseNet121 + CBAM"],
      ["Report ID", `RPT-${date.toISOString().slice(0, 10).replace(/-/g, "")}-${Date.now().toString(36).slice(-4).toUpperCase()}`],
    ],
  });

  y = getFinalY(doc, y);

  y = addSectionTitle(doc, "Classification summary", y);

  autoTable(doc, {
    ...defaultTableStyles,
    startY: y,
    tableWidth: CONTENT_WIDTH,
    head: [["Field", "Value"]],
    body: [
      ["Predicted class", predicted_class ?? "—"],
      ["Description", CLASS_LABELS[predicted_class] ?? "—"],
      ["Confidence", confidence != null ? `${Number(confidence).toFixed(2)}%` : "—"],
    ],
    columnStyles: {
      0: { cellWidth: 52, fontStyle: "bold" },
      1: { cellWidth: CONTENT_WIDTH - 52 },
    },
    didParseCell: (data) => {
      if (data.section === "body" && data.row.index === 0) {
        data.cell.styles.fillColor = COLORS.predicted;
        data.cell.styles.fontStyle = "bold";
      }
    },
  });

  y = getFinalY(doc, y);

  y = addSectionTitle(doc, "Class probabilities", y);

  const probRows = CLASSES.map((cls) => {
    const prob = probabilities?.[cls];
    const isPredicted = cls === predicted_class;
    return [
      cls,
      CLASS_LABELS[cls] ?? "—",
      prob != null ? Number(prob).toFixed(2) : "—",
      isPredicted ? "Yes" : "—",
    ];
  });

  autoTable(doc, {
    ...defaultTableStyles,
    startY: y,
    tableWidth: CONTENT_WIDTH,
    head: [["Class", "Description", "Probability (%)", "Predicted"]],
    body: probRows,
    columnStyles: {
      0: { cellWidth: 45 },
      1: { cellWidth: 70 },
      2: { cellWidth: 40, halign: "right" },
      3: { cellWidth: 27, halign: "center" },
    },
    didParseCell: (data) => {
      if (data.section === "body" && data.row.raw[3] === "Yes") {
        data.cell.styles.fillColor = COLORS.predicted;
        if (data.column.index === 0 || data.column.index === 2) {
          data.cell.styles.fontStyle = "bold";
        }
      }
    },
  });

  y = getFinalY(doc, y);

  // --- Images ---
  const hasOriginal = Boolean(originalPreview);
  const hasGradcam = Boolean(gradcamBase64);

  if (hasOriginal || hasGradcam) {
    if (y > 200) {
      doc.addPage();
      y = MARGIN;
    }

    y = addSectionTitle(doc, "Imaging", y);

    const imgSize = hasOriginal && hasGradcam ? 82 : 90;
    const gap = 8;
    let xOffset = MARGIN;

    if (hasOriginal) {
      const origImg = await loadImage(originalPreview);
      const origData = compressImage(origImg, 400);

      doc.setFont("helvetica", "bold");
      doc.setFontSize(9);
      doc.setTextColor(...COLORS.primary);
      doc.text("Original MRI", xOffset, y);

      doc.setDrawColor(...COLORS.border);
      doc.setLineWidth(0.2);
      doc.rect(xOffset - 1, y + 2, imgSize + 2, imgSize + 2);

      doc.addImage(origData, "JPEG", xOffset, y + 4, imgSize, imgSize);
      xOffset += imgSize + gap + 4;
    }

    if (hasGradcam) {
      const gradcamSrc = gradcamBase64.startsWith("data:")
        ? gradcamBase64
        : `data:image/png;base64,${gradcamBase64}`;
      const gradImg = await loadImage(gradcamSrc);
      const gradData = compressImage(gradImg, 400);

      doc.setFont("helvetica", "bold");
      doc.setFontSize(9);
      doc.setTextColor(...COLORS.primary);
      doc.text("Grad-CAM overlay", xOffset, y);

      doc.setDrawColor(...COLORS.border);
      doc.rect(xOffset - 1, y + 2, imgSize + 2, imgSize + 2);

      doc.addImage(gradData, "JPEG", xOffset, y + 4, imgSize, imgSize);
    }

    y += imgSize + 14;
  }

  addFooter(doc);

  const pdfBlob = doc.output("blob");
  const maxSize = 5 * 1024 * 1024;

  if (pdfBlob.size > maxSize) {
    throw new Error("PDF_TOO_LARGE");
  }

  const fileName = `alzheimer_report_${date.toISOString().slice(0, 10)}.pdf`;
  const url = URL.createObjectURL(pdfBlob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};
