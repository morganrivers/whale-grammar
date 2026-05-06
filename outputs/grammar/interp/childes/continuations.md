# Continuations — childes

Sampling: coda head `T=0.9, top-k=40`; DT head `T=0.9, no top-k` (3 buckets). Mean held-out seed length is used as the target continuation length.

## seed 0 — `childes::020700::5::18`
original length: 100; generating 87 new tokens after the K=8 prefix

**prefix (first 8 tokens, rendered from real CSV):**
```
  *CHI:	i get it .
  *MOT:	oh no .
  *MOT:	wait .
  *MOT:	wait .
  *MOT:	wait .
```

**original continuation (ground truth):**
```
  *CHI:	you want to be a monster .
  *INV:	pardon .
  *CHI:	you want to be a monster .
  *MOT:	oh .
  *MOT:	hey .
  *MOT:	you be get a po cm gail .
  *MOT:	be that good .
  *MOT:	for andy .
  *MOT:	be super cm will not it .
  *INV:	can not real see .
  *MOT:	caroline be mean to be write cm darling .
  *INV:	smell nice though .
  *CHI:	andy want to .
  *CHI:	you want to be a monster .
  *MOT:	let us smell .
  *CHI:	tie some round your arm .
  *MOT:	oh .
  *MOT:	what be happen to my arm .
  *MOT:	oh dear .
  *MOT:	something be happen to it .
  *MOT:	it be go all sort of flop cm .
```

**sample 1:**
```
  *CHI:	your us can this a you you go go yeah that go thank go down it .
  *MOT:	a a this in this want you bit yeah a .
  *INV:	be bit be be oh .
  *INV:	a .
  *INV:	there be go .
  *CHI:	to yeah and good be get .
  *CHI:	do you play be be time be right .
  *CHI:	sauce .
  *CHI:	be this a be .
  *MOT:	not right hm .
  *MOT:	no not truck now what be .
  *INV:	a .
  *INV:	in you .
  *CHI:	bang be be they the .
  *CHI:	go have you be quite a be be go be .
  *CHI:	not oh be be just .
  *CHI:	your .
```

**sample 2:**
```
  *CHI:	little bit be cm then i this be the .
  *CHI:	a .
  *CHI:	you what think .
  *CHI:	we it to go .
  *MOT:	go be do .
  *INV:	sleep have do oh .
  *INV:	this not .
  *INV:	do .
  *CHI:	not know have be now they a .
  *MOT:	go to go cm be go .
  *INV:	oh be .
  *CHI:	have oh you there .
  *MOT:	it it on .
  *INV:	do not be do it can .
  *INV:	not three bit be i the go .
  *INV:	to put my be .
  *INV:	be this just a .
  *INV:	end have .
  *INV:	be .
  *CHI:	oh .
  *CHI:	that be .
  *CHI:	a .
  *CHI:	oh have look .
  *CHI:	yeah .
  *CHI:	a go no be .
  *CHI:	be all .
```

**sample 3:**
```
  *CHI:	your on get she eat one look at a on .
  *MOT:	be cm want to what do this you do be baby sleep be not not and get .
  *INV:	to come be the end go be you on do not end we me have .
  *CHI:	have and .
  *CHI:	do .
  *CHI:	have .
  *MOT:	in .
  *INV:	do get end look no that .
  *INV:	there bit it be go not have to a a .
  *INV:	bit to not do want .
  *CHI:	end a one you want have you a and a be a be not the .
  *CHI:	go a be with .
```

## seed 1 — `childes::021106b::8::55`
original length: 103; generating 87 new tokens after the K=8 prefix

**prefix (first 8 tokens, rendered from real CSV):**
```
  *FAT:	yes cm my darling .
  *CHI:	how do you switch .
```

**original continuation (ground truth):**
```
  *CHI:	this on .
  *FAT:	i think you might have switch it on already .
  *FAT:	actual .
  *FAT:	now what we do be plug this in at the back and let us have a look .
  *CHI:	so what be that noise cm dad .
  *FAT:	that be the cd work .
  *FAT:	hold on .
  *FAT:	you know if you sit up here and i will just see if i can get it to work .
  *FAT:	there you go .
  *FAT:	right .
  *FAT:	and then to switch it on it be this button here .
  *FAT:	there it be .
  *FAT:	that be a party one again .
  *FAT:	you .
```

**sample 1:**
```
  *FAT:	and go know to be old want to play .
  *CHI:	can be do all on yeah be here .
  *FAT:	not i go yeah .
  *FAT:	this .
  *CHI:	not it your .
  *FAT:	in this it get a there that have get end .
  *FAT:	a .
  *FAT:	bit cm .
  *CHI:	be a it will one be the one cm darling to do not one middle get and have be be can my get oh go do not on put be a not then this this do .
  *CHI:	be us bit .
  *FAT:	get not cm be not cm be what .
  *CHI:	this the .
```

**sample 2:**
```
  *FAT:	i go .
  *FAT:	yeah and have to a and say bit be cm .
  *CHI:	be just .
  *FAT:	eat know go be say yeah the what .
  *FAT:	can you .
  *CHI:	you bit you .
  *FAT:	be what up want do have on .
  *CHI:	it .
  *CHI:	you .
  *FAT:	be it you one .
  *CHI:	where do know that .
  *CHI:	be yeah cm not .
  *CHI:	mummy .
  *CHI:	you .
  *FAT:	do do not .
  *CHI:	it a do yeah have .
  *CHI:	not we cm all good .
  *CHI:	look they be .
  *FAT:	have and then what have .
  *CHI:	go go it .
  *CHI:	your to it end be .
  *FAT:	alright will be silly .
  *FAT:	want .
  *FAT:	be not be .
```

**sample 3:**
```
  *FAT:	it do look .
  *FAT:	you be be go say not one have go a cm get come me do to be .
  *FAT:	can do the yeah what he .
  *FAT:	one look you want you be .
  *CHI:	get .
  *CHI:	you your and it have come .
  *CHI:	what .
  *FAT:	and where go be do it .
  *CHI:	look be .
  *CHI:	not .
  *CHI:	one to .
  *CHI:	that be .
  *FAT:	be be .
  *FAT:	go be alright think there not want bus a .
  *FAT:	in .
  *FAT:	be .
  *CHI:	i you be and what not sit be go do look .
  *FAT:	you .
  *CHI:	be go to your not get go time of .
```

## seed 2 — `childes::021106b::8::19`
original length: 82; generating 87 new tokens after the K=8 prefix

**prefix (first 8 tokens, rendered from real CSV):**
```
  *CHI:	those toy .
  *FAT:	oh yes .
  *FAT:	so they pretend to .
```

**original continuation (ground truth):**
```
  *FAT:	eat them all .
  *FAT:	they do not real eat .
  *FAT:	we eat end do not we .
  *FAT:	we can have a picnic .
  *FAT:	what you do .
  *CHI:	i be get the fish finger in here .
  *FAT:	oh .
  *FAT:	very nice .
  *CHI:	fish finger .
  *FAT:	oh ooh .
  *CHI:	do you like fish finger .
  *FAT:	go and wash your hand cm darling .
  *CHI:	yes .
  *FAT:	ah .
  *FAT:	wash .
  *FAT:	oh .
  *FAT:	get tissue from the .
  *FAT:	you have get too much sticky on there .
  *CHI:	so clean .
  *FAT:	no .
  *FAT:	it be not real still sticky .
```

**sample 1:**
```
  *CHI:	oh dear my .
  *CHI:	what one with this in say be be in .
  *FAT:	you what have there that it go to go you it no some be i can a be to .
  *CHI:	look big quite the .
  *CHI:	be one .
  *CHI:	what .
  *FAT:	cm mummy will if boy i be in this want can you .
  *FAT:	there bit be you can have go yeah a in look have be be do not be on .
  *FAT:	look .
  *FAT:	a be .
  *FAT:	good .
  *FAT:	this look a .
  *FAT:	be there get of go you you a be oh .
  *FAT:	can mummy .
```

**sample 2:**
```
  *CHI:	be in this baby look .
  *CHI:	the what it it be a be go .
  *CHI:	be .
  *CHI:	go oh oh .
  *CHI:	one mummy look cm be .
  *CHI:	go go what go you to do you i go can on .
  *CHI:	look go to end will come do look yeah i can do you they you .
  *CHI:	be go .
  *CHI:	it be will have mummy bit a bit go the that that we .
  *FAT:	us do be us what be my .
  *FAT:	not cm cm my .
  *CHI:	a you it be .
  *FAT:	a not we you be you be not .
```

**sample 3:**
```
  *CHI:	be the not go go .
  *CHI:	be the look bear have go to what sit go see do .
  *FAT:	to do i yeah in go will go what get be two be .
  *CHI:	go .
  *FAT:	do be not a in be time your think .
  *CHI:	yeah go he what what be this do you it your .
  *FAT:	shall be go .
  *CHI:	and be you be one and .
  *FAT:	i a you go do want want not .
  *CHI:	one .
  *FAT:	to and and on be in the .
  *FAT:	have will yeah that be .
  *FAT:	not be go and go good .
```
