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

	load = function(VOCdevkit_VOCYEAR)
		local ffi = require 'ffi'
		local xml = require 'xml'

		local filelists = 
		{
			train = paths.concat(VOCdevkit_VOCYEAR, 'ImageSets/Main/train.txt'),
			val	= paths.concat(VOCdevkit_VOCYEAR, 'ImageSets/Main/val.txt'),
			test = paths.concat(VOCdevkit_VOCYEAR, 'ImageSets/Main/test.txt'),
		}

		local numMaxSamples = 11000
		local numMaxObjectsPerSample = 5

		local mkDataset = function() return 
		{
			filenames = torch.CharTensor(numMaxSamples, 16):zero(),
			labels = torch.FloatTensor(numMaxSamples, #classLabels):zero(),
			objectBoxes = torch.FloatTensor(numMaxSamples * numMaxObjectsPerSample, 5):zero(),
			objectBoxesInds = torch.IntTensor(numMaxSamples, 2):zero(),
			jpegs = torch.ByteTensor(numMaxSamples * 3 * 50000):zero(),
			jpegsInds = torch.IntTensor(numMaxSamples, 2):zero(),

			getNumSamples = function(self)
				return self.numSamples
			end,

			getImageFileName = function(self, exampleIdx)
				return (require 'ffi').string(self.filenames[exampleIdx]:data())
			end,

			getGroundTruthBoxes = function(self, exampleIdx)
				return self.objectBoxes:sub(self.objectBoxesInds[exampleIdx][1], self.objectBoxesInds[exampleIdx][2])
			end,

			getJpegBytes = function(self, exampleIdx)
				return self.jpegs:sub(self.jpegsInds[exampleIdx][1], self.jpegsInds[exampleIdx][2])
			end,

			getLabels = function(self, exampleIdx)
				return self.labels[exampleIdx]
			end
		} end

		local voc = { train = mkDataset(), val = mkDataset(), test = mkDataset() }

		for _, subset in ipairs{'train', 'val', 'test'} do
			local exampleIdx = 1
			local jpegsFirstByteInd = 1
			for line in io.lines(filelists[subset]) do
				assert(exampleIdx <= numMaxSamples)
				assert(#line < voc[subset].filenames:size(2))

				ffi.copy(voc[subset].filenames[exampleIdx]:data(), line)
					
				local f = torch.DiskFile(paths.concat(VOCdevkit_VOCYEAR, 'JPEGImages', line .. '.jpg'), 'r')
				f:binary()
				f:seekEnd()
				local file_size_bytes = f:position() - 1
				f:seek(1)
				local bytes = torch.ByteTensor(file_size_bytes)
				f:readByte(bytes:storage())
				voc[subset].jpegsInds[exampleIdx] = torch.IntTensor({jpegsFirstByteInd, jpegsFirstByteInd + file_size_bytes - 1})
				voc[subset]:getJpegBytes(exampleIdx):copy(bytes)
				f:close()

				jpegsFirstByteInd = voc[subset].jpegsInds[exampleIdx][2] + 1
				exampleIdx = exampleIdx + 1
			end
			voc[subset].numSamples = exampleIdx - 1
		end	 
		local testHasAnnotation = VOCdevkit_VOCYEAR:find('2007') ~= nil
		for _, subset in ipairs(testHasAnnotation and {'train', 'val', 'test'} or {'train', 'val'})  do
			for classLabelInd, v in ipairs(classLabels) do
				local exampleIdx = 1
				for line in io.lines(paths.concat(VOCdevkit_VOCYEAR, 'ImageSets/Main/'..v..'_'..subset..'.txt')) do
					if string.find(line, ' -1', 1, true) then
						voc[subset].labels[exampleIdx][classLabelInd] = -1
					elseif string.find(line, ' 1', 1, true) then
						voc[subset].labels[exampleIdx][classLabelInd] = 1
					end
					exampleIdx = exampleIdx + 1
				end
			end

			local exampleIdx = 1
			local objectBoxIdx = 1
			for line in io.lines(filelists[subset]) do
				local anno_xml = xml.loadpath(paths.concat(VOCdevkit_VOCYEAR, 'Annotations/' .. line ..'.xml'))

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
				
				voc[subset].objectBoxesInds[exampleIdx] = torch.IntTensor({firstObjectBoxIdx, objectBoxIdx - 1})
				exampleIdx = exampleIdx + 1
			end
		end

		if not testHasAnnotation then
			voc['test'].objectBoxesInds = nil
			voc['test'].objectBoxes = nil
		end
		
		for _, subset in ipairs{'train', 'val', 'test'} do
			voc[subset].filenames = voc[subset].filenames:sub(1, voc[subset].numSamples)
			voc[subset].labels = voc[subset].labels:sub(1, voc[subset].numSamples)
			voc[subset].jpegsInds = voc[subset].jpegsInds:sub(1, voc[subset].numSamples)
			voc[subset].jpegs = voc[subset].jpegs:sub(1, voc[subset].jpegsInds[voc[subset].numSamples][2])

			if voc[subset].objectBoxes and voc[subset].objectBoxesInds then
				voc[subset].objectBoxesInds =  voc[subset].objectBoxesInds:sub(1, voc[subset].numSamples)
				voc[subset].objectBoxes = voc[subset].objectBoxes:sub(1, voc[subset].objectBoxesInds[voc[subset].numSamples][2])
			end
		end

		voc['trainval'] = {
			getNumSamples = function(self)
				return voc['train']:getNumSamples() + voc['val']:getNumSamples()
			end,

			getImageFileName = function(self, exampleIdx)
				return exampleIdx <= voc['train']:getNumSamples() and voc['train']:getImageFileName(exampleIdx) or voc['val']:getImageFileName(exampleIdx - voc['train']:getNumSamples())
			end,

			getGroundTruthBoxes = function(self, exampleIdx)
				return exampleIdx <= voc['train']:getNumSamples() and voc['train']:getGroundTruthBoxes(exampleIdx) or voc['val']:getGroundTruthBoxes(exampleIdx - voc['train']:getNumSamples())
			end,

			getJpegBytes = function(self, exampleIdx)
				return exampleIdx <= voc['train']:getNumSamples() and voc['train']:getJpegBytes(exampleIdx) or voc['val']:getJpegBytes(exampleIdx - voc['train']:getNumSamples())
			end,

			getLabels = function(self, exampleIdx)
				return exampleIdx <= voc['train']:getNumSamples() and voc['train']:getLabels(exampleIdx) or voc['val']:getLabels(exampleIdx - voc['train']:getNumSamples())
			end
		}

		return voc
	end,

	package_submission = function(OUT, voc, VOCYEAR, subset, task, ...)
		local task_a, task_b  = task:sub(1, 4), task:sub(6, 8)
		local write = {
			cls = function(f, classLabelInd, scores)
				if voc[subset].numSamples ~= scores:size(1) then
					print('WARNING: scores is not full size: ', 'expected:', voc[subset].numSamples, 'actual:', scores:size(1))
				end
				scores = scores:select(2, classLabelInd)

				for i = 1, voc[subset].numSamples do
					f:write(string.format('%s %.12f\n', voc[subset]:getImageFileName(i), scores[i]))
				end
			end,
			det = function(f, classLabelInd, scores, rois, mask)
				if voc[subset].numSamples ~= scores:size(1) or voc[subset].numSamples ~= rois:size(1) then
					print('WARNING: scores or rois are not full size: ', 'expected:', voc[subset].numSamples, 'bad:', scores:size(1), rois:size(1))
				end
				
				local inds = mask:select(2, classLabelInd):nonzero()
				for i = 1, inds:size(1) do
					local exampleIdx, roiInd = unpack(inds[i]:totable())
					f:write(string.format('%s %.12f %.12f %.12f %.12f %.12f\n', voc[subset]:getImageFileName(exampleIdx), scores[exampleIdx][classLabelInd][roiInd], unpack(rois[exampleIdx][roiInd]:totable())))
				end
			end
		}

		os.execute(string.format('rm -rf %s/results', OUT))
		os.execute(string.format('mkdir -p %s/results/%s/Main', OUT, VOCYEAR))

		local respath = string.format('%s/results/%s/Main/%%s_%s_%s_%%s.txt', OUT, VOCYEAR, task_b, subset)
		for classLabelInd, classLabel in ipairs(classLabels) do
			local f = assert(io.open(respath:format(task_a, classLabel), 'w'))
			write[task_b](f, classLabelInd, ...)
			f:close()
		end
		os.execute(string.format('cd %s && tar -czf results-%s-%s-%s.tar.gz results', OUT, VOCYEAR, task, subset))
		return respath
	end,

	vis_classification_submission = function(OUT, VOCYEAR, subset, classLabel, JPEGImages_DIR, top_k)
		top_k = top_k or 20
		local res_file_path = string.format('%s/results/%s/Main/comp2_cls_%s_%s.txt', OUT, VOCYEAR, subset, classLabel)

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
		return ({precisionrecall(scores_all, labels_all)})[3]:mean()
	end
}
