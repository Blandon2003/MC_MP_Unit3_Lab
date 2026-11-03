$ErrorActionPreference = "Stop"
$docx = "C:\Users\AH THEC\Documents\document\computer engineering level 3\semester 2\Mobile application\acoustiguard (1)\MC_MP_Unit3_Lab\MC_MP_Unit3_Lab_Report.docx"
$pdf  = "C:\Users\AH THEC\Documents\document\computer engineering level 3\semester 2\Mobile application\acoustiguard (1)\MC_MP_Unit3_Lab\MC_MP_Unit3_Lab_Report.pdf"

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$doc = $word.Documents.Open($docx, $false, $false)

Write-Output ("TOC_COUNT=" + $doc.TablesOfContents.Count)
try {
    if ($doc.TablesOfContents.Count -gt 0) {
        $doc.TablesOfContents.Item(1).Update() | Out-Null
        Write-Output "TOC_UPDATED=OK"
    }
} catch {
    Write-Output ("TOC_UPDATE_FAILED=" + $_.Exception.Message)
}
try { $doc.Fields.Update() | Out-Null } catch { Write-Output ("FIELDS_FAILED=" + $_.Exception.Message) }
$doc.Repaginate()
$doc.Save() | Out-Null
$doc.ExportAsFixedFormat($pdf, 17)
Write-Output ("PAGES=" + $doc.ComputeStatistics(2))
$doc.Close(0)
$word.Quit()
Write-Output "PDF_EXPORTED=$pdf"
