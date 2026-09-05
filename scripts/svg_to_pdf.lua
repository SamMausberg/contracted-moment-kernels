-- Use vector PDF siblings for LaTeX, retaining SVG links in GitHub Markdown.
function Image(image)
  if FORMAT:match("latex") then
    image.src = image.src:gsub("%.svg$", ".pdf")
    image.attributes.width = "100%"
  end
  return image
end

-- A downloaded PDF should retain working links to the repository's evidence.
function Link(link)
  if FORMAT:match("latex") and not link.target:match("^%a[%w+.-]*:")
      and not link.target:match("^#") then
    local target = link.target
    if target:match("^%.%./") then
      target = target:gsub("^%.%./", "")
    else
      target = "paper/" .. target
    end
    link.target = "https://github.com/SamMausberg/contracted-moment-kernels/blob/main/" .. target
  end
  return link
end

-- Keep each scientific figure with its explanatory caption; avoid duplicate
-- automatic captions and floating images separated from the manuscript text.
function Pandoc(document)
  if not FORMAT:match("latex") then return document end
  local blocks = {}
  local index = 1
  while index <= #document.blocks do
    local block = document.blocks[index]
    local following = document.blocks[index + 1]
    if block.t == "Para" and #block.content == 1 and block.content[1].t == "Image"
        and following and following.t == "Para" then
      table.insert(blocks, pandoc.RawBlock("latex", "\\begin{minipage}{\\linewidth}"))
      table.insert(blocks, block)
      table.insert(blocks, following)
      table.insert(blocks, pandoc.RawBlock("latex", "\\end{minipage}\\par\\medskip"))
      index = index + 2
    else
      table.insert(blocks, block)
      index = index + 1
    end
  end
  document.blocks = blocks
  return document
end
