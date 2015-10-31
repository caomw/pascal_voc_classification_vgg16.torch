local classLabels = {'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'}

local function precisionrecall(scores_all, labels_all)
	--adapted from VOCdevkit/VOCcode/VOCevalcls.m (VOCap.m). tested, gives equivalent results
	local function VOCap(rec, prec)
		local mrec = torch.cat(torch.cat(torch.FloatTensor({0}), rec), torch.FloatTensor({1}))
		local mpre = torch.cat(torch.cat(torch.FloatTensor({0}), prec), torch.FloatTensor({0}))
		for i=mpre:numel()-1, 1, -1 do
			mpre[i]=math.max(mpre[i], mpre[i+1])
		end

		local i = (mrec:sub(2, mrec:numel())):ne(mrec:sub(1, mrec:numel() - 1)):nonzero():squeeze(2) + 1
		local ap = (mrec:index(1, i) - mrec:index(1, i - 1)):cmul(mpre:index(1, i)):sum()

		return ap
	end

	local function VOCevalcls(out, gt)
		local so,si= (-out):sort()

		local tp=gt:index(1, si):gt(0):float()
		local fp=gt:index(1, si):lt(0):float()

		fp=fp:cumsum()
		tp=tp:cumsum()

		local rec=tp/gt:gt(0):sum()
		local prec=tp:cdiv(fp+tp)

		local ap=VOCap(rec,prec)
		return rec, prec, ap
	end

	local prec = torch.FloatTensor(scores_all:size())
	local rec = torch.FloatTensor(scores_all:size())
	local ap = torch.FloatTensor(#classLabels)

	for classLabelInd = 1, #classLabels do
		local p, r, a = VOCevalcls(scores_all:narrow(2, classLabelInd, 1):squeeze(), labels_all:narrow(2, classLabelInd, 1):squeeze())
		prec:narrow(2, classLabelInd, 1):copy(p)
		rec:narrow(2, classLabelInd, 1):copy(r)
		ap[classLabelInd] = a
	end

	return prec, rec, ap
end

return {
	classLabels = classLabels,
	numClasses = #classLabels,

	load = function(trainval_VOCdevkit_VOC2012, test_VOCdevkit_VOC2012, read_box_annotation)
		local ffi = require 'ffi'
		local tds = require 'tds'
		local xml = require 'xml'

		local filelists = 
		{
			train = paths.concat(trainval_VOCdevkit_VOC2012, 'ImageSets/Main/train.txt'),
			val	= paths.concat(trainval_VOCdevkit_VOC2012, 'ImageSets/Main/val.txt'),
			test = paths.concat(test_VOCdevkit_VOC2012, 'ImageSets/Main/test.txt'),
		}

		local numMaxSamples = 11000
		local numMaxObjectsPerSample = 5

		local mkDataset = function() return 
		{
			filenames = torch.CharTensor(numMaxSamples, 16):zero(),
			labels = torch.FloatTensor(numMaxSamples, #classLabels):zero(),
			objectBoxes = torch.FloatTensor(numMaxSamples * numMaxObjectsPerSample, 5):zero(),
			objectBoxesInds = torch.FloatTensor(numMaxSamples, 2):zero(),
			jpegs = tds.hash(),
			getFileName = function(self, exampleIdx)
				return (require 'ffi').string(self.filenames[exampleIdx]:data())
			end,
			getBoxes = function(self, exampleIdx)
				return self.objectBoxes:narrow(1, self.objectBoxesInds[exampleIdx][1], self.objectBoxesInds[exampleIdx][2])
			end
		} end

		local voc = {train = mkDataset(), val = mkDataset(), test = mkDataset()}

		for _, subset in ipairs{'train', 'val', 'test'} do
			local exampleIdx = 1
			for line in io.lines(filelists[subset]) do
				assert(exampleIdx <= numMaxSamples)
				assert(#line < voc[subset].filenames:size(2))

				ffi.copy(voc[subset].filenames:data() + voc[subset].filenames:size(2) * (exampleIdx - 1), line)
					
				local f = torch.DiskFile(paths.concat(subset == 'test' and test_VOCdevkit_VOC2012 or trainval_VOCdevkit_VOC2012, 'JPEGImages', line .. '.jpg'), 'r')
				f:binary()
				f:seekEnd()
				local file_size_bytes = f:position() - 1
				f:seek(1)
				voc[subset].jpegs[exampleIdx] = torch.ByteTensor(file_size_bytes)
				f:readByte(voc[subset].jpegs[exampleIdx]:storage())
				f:close()

				exampleIdx = exampleIdx + 1
			end
		end	 

		for _, subset in ipairs{'train', 'val'} do
			for classLabelInd, v in ipairs(classLabels) do
				local exampleIdx = 1
				for line in io.lines(paths.concat(trainval_VOCdevkit_VOC2012, 'ImageSets/Main/'..v..'_'..subset..'.txt')) do
					if string.find(line, ' -1', 1, true) then
						voc[subset].labels[exampleIdx][classLabelInd] = -1
					elseif string.find(line, ' 1', 1, true) then
						voc[subset].labels[exampleIdx][classLabelInd] = 1
					end
					exampleIdx = exampleIdx + 1
				end
			end

			if read_box_annotation then
				local exampleIdx = 1
				local objectBoxIdx = 1
				for line in io.lines(filelists[subset]) do
					local anno_xml = xml.loadpath(paths.concat(trainval_VOCdevkit_VOC2012, 'Annotations/' .. line ..'.xml'))

					local firstObjectBoxIdx = objectBoxIdx
					for i = 1, #anno_xml do
						if anno_xml[i].xml == 'object' then
							local classLabel = xml.find(anno_xml[i], 'name')[1]
							local xmin = xml.find(xml.find(anno_xml[i], 'bndbox'), 'xmin')[1]
							local xmax = xml.find(xml.find(anno_xml[i], 'bndbox'), 'xmax')[1]
							local ymin = xml.find(xml.find(anno_xml[i], 'bndbox'), 'ymin')[1]
							local ymax = xml.find(xml.find(anno_xml[i], 'bndbox'), 'ymax')[1]

							for classLabelInd = 1, #classLabels do
								if classLabels[classLabelInd] == classLabel then
									assert(objectBoxIdx <= voc[subset].objectBoxes:size(1))

									voc[subset].objectBoxes[objectBoxIdx] = torch.FloatTensor({classLabelInd, xmin, ymin, xmax, ymax})
									objectBoxIdx = objectBoxIdx + 1
								end
							end
						end
					end
					
					voc[subset].objectBoxesInds[exampleIdx] = torch.FloatTensor({firstObjectBoxIdx, objectBoxIdx - firstObjectBoxIdx})
					exampleIdx = exampleIdx + 1
				end
			end
		end
		
		for _, subset in ipairs{'train', 'val', 'test'} do
			voc[subset].numSamples = #voc[subset].jpegs
			voc[subset].filenames = voc[subset].filenames:narrow(1, 1, voc[subset].numSamples)

			if subset ~= 'test' then
				voc[subset].labels = voc[subset].labels:narrow(1, 1, voc[subset].numSamples)
			else
				voc[subset].labels = nil
			end

			if subset ~= 'test' and read_box_annotation then
				voc[subset].objectBoxesInds =  voc[subset].objectBoxesInds:narrow(1, 1, voc[subset].numSamples)
				voc[subset].objectBoxes = voc[subset].objectBoxes:narrow(1, 1, voc[subset].objectBoxesInds[voc[subset].numSamples][1] + voc[subset].objectBoxesInds[voc[subset].numSamples][2])
			else
				voc[subset].objectBoxesInds = nil	
				voc[subset].objectBoxes = nil
			end
		end

		return voc
	end,

	package_submission = function(OUT, voc, subset, task, ...)
		local write = {
			comp2_cls = function(f, classLabelInd, scores)
				if voc[subset].numSamples ~= scores:size(1) then
					print('WARNING: scores is not full size: ', 'expected:', voc[subset].numSamples, 'actual:', scores:size(1))
				end
				scores = scores:select(2, classLabelInd)

				for i = 1, voc[subset].numSamples do
					f:write(string.format('%s %.12f\n', voc[subset]:getFileName(i), scores[i]))
				end
			end,
			comp4_det = function(f, classLabelInd, scores, rois, keep)
				if voc[subset].numSamples ~= scores:size(1) or voc[subset].numSamples ~= rois:size(1) then
					print('WARNING: scores or rois are not full size: ', 'expected:', voc[subset].numSamples, 'bad:', scores:size(1), rois:size(1))
				end

				for i = 1, keep:size(1) do
					for j = 1, keep:size(2) do
						local roiInd = keep[i][j][classLabelInd]
						if roiInd == 0 then
							break
						end
						f:write(string.format('%s %.12f %.12f %.12f %.12f %.12f\n', voc[subset]:getFileName(i), scores[i][roiInd][classLabelInd], rois[i][roiInd][1], rois[i][roiInd][2], rois[i][roiInd][3], rois[i][roiInd][4]))
					end
				end
			end
		}

		os.execute(string.format('rm -rf %s/results', OUT))
		os.execute(string.format('mkdir -p %s/results/VOC2012/Main', OUT))
		for classLabelInd, classLabel in ipairs(classLabels) do
			local f = assert(io.open(string.format('%s/results/VOC2012/Main/%s_%s_%s.txt', OUT, task, subset, classLabel), 'w'))
			write[task](f, classLabelInd, ...)
			f:close()
		end
		os.execute(string.format('cd %s && tar -czf results-voc2012-%s-%s.tar.gz results', OUT, task, subset))
	end,

	vis_classification_submission = function(OUT, subset, classLabel, JPEGImages_DIR, top_k)
		top_k = top_k or 20
		local res_file_path = string.format('%s/results/VOC2012/Main/comp2_cls_%s_%s.txt', OUT, subset, classLabel)

		local scores = {}
		for line in assert(io.open(res_file_path)):lines() do
			scores[#scores + 1] = line:split(' ')
		end

		table.sort(scores, function(a, b) return -tonumber(a[2]) < -tonumber(b[2]) end)

		local image = require 'image'
		local top_imgs = {}
		print('K = ', top_k)
		for i = 1, top_k do
			top_imgs[i] = image.scale(image.load(paths.concat(JPEGImages_DIR, scores[i][1] .. '.jpg')), 128, 128)
			print(scores[i][2], scores[i][1])
		end

		image.display(top_imgs)
	end,
	
	precisionrecall = precisionrecall,

	meanAP = function(scores_all, labels_all)
		return ({precisionrecall(scores_all, labels_all)})[3]
	end
}
