let SessionLoad = 1
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~/Notes/AI+ML/10703/hw3/p2-templates
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
argglobal
%argdel
$argadd model_pytorch.py
edit imitation.py
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
split
wincmd _ | wincmd |
split
2wincmd k
wincmd w
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
wincmd w
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe '1resize ' . ((&lines * 14 + 22) / 45)
exe '2resize ' . ((&lines * 15 + 22) / 45)
exe 'vert 2resize ' . ((&columns * 111 + 96) / 193)
exe '3resize ' . ((&lines * 15 + 22) / 45)
exe 'vert 3resize ' . ((&columns * 81 + 96) / 193)
exe '4resize ' . ((&lines * 10 + 22) / 45)
exe 'vert 4resize ' . ((&columns * 96 + 96) / 193)
exe '5resize ' . ((&lines * 10 + 22) / 45)
exe 'vert 5resize ' . ((&columns * 96 + 96) / 193)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=4
setlocal fml=1
setlocal fdn=4
setlocal fen
14
normal! zo
61
normal! zo
80
normal! zo
102
normal! zo
118
normal! zo
145
normal! zo
let s:l = 3 - ((2 * winheight(0) + 7) / 14)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 3
normal! 0
wincmd w
argglobal
if bufexists("imitation.py") | buffer imitation.py | else | edit imitation.py | endif
if &buftype ==# 'terminal'
  silent file imitation.py
endif
balt model_pytorch.py
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=4
setlocal fml=1
setlocal fdn=4
setlocal fen
14
normal! zo
61
normal! zo
80
normal! zo
102
normal! zo
118
normal! zo
145
normal! zo
let s:l = 139 - ((6 * winheight(0) + 7) / 15)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 139
normal! 0
wincmd w
argglobal
if bufexists("BCDAGGER.py") | buffer BCDAGGER.py | else | edit BCDAGGER.py | endif
if &buftype ==# 'terminal'
  silent file BCDAGGER.py
endif
balt utils.py
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=4
setlocal fml=1
setlocal fdn=4
setlocal fen
32
normal! zo
38
normal! zo
39
normal! zo
let s:l = 46 - ((9 * winheight(0) + 7) / 15)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 46
normal! 0
wincmd w
argglobal
if bufexists("playground.py") | buffer playground.py | else | edit playground.py | endif
if &buftype ==# 'terminal'
  silent file playground.py
endif
balt ~/Notes/AI+ML/10703/hw3/playground.py
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=2
setlocal fml=1
setlocal fdn=4
setlocal fen
let s:l = 33 - ((5 * winheight(0) + 5) / 10)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 33
normal! 0
wincmd w
argglobal
if bufexists("term://~/Notes/AI+ML/10703/hw3/p2-templates//131602:/usr/bin/ipython") | buffer term://~/Notes/AI+ML/10703/hw3/p2-templates//131602:/usr/bin/ipython | else | edit term://~/Notes/AI+ML/10703/hw3/p2-templates//131602:/usr/bin/ipython | endif
if &buftype ==# 'terminal'
  silent file term://~/Notes/AI+ML/10703/hw3/p2-templates//131602:/usr/bin/ipython
endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=4
setlocal fen
let s:l = 10001 - ((3 * winheight(0) + 5) / 10)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 10001
normal! 0
wincmd w
2wincmd w
exe '1resize ' . ((&lines * 14 + 22) / 45)
exe '2resize ' . ((&lines * 15 + 22) / 45)
exe 'vert 2resize ' . ((&columns * 111 + 96) / 193)
exe '3resize ' . ((&lines * 15 + 22) / 45)
exe 'vert 3resize ' . ((&columns * 81 + 96) / 193)
exe '4resize ' . ((&lines * 10 + 22) / 45)
exe 'vert 4resize ' . ((&columns * 96 + 96) / 193)
exe '5resize ' . ((&lines * 10 + 22) / 45)
exe 'vert 5resize ' . ((&columns * 96 + 96) / 193)
tabnext 1
badd +124 imitation.py
badd +89 ~/Notes/AI+ML/10703/hw2/hw2_code/pytorch/a2c/a2c.py
badd +0 ~/Notes/AI+ML/10703/hw3/playground
badd +9 ~/Notes/AI+ML/10703/hw3/playground.py
badd +109 BCDAGGER.py
badd +72 model_pytorch.py
badd +25 playground_done.py
badd +21 playground.py
badd +30 utils.py
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0 && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 shortmess=filnxtToOFc
let &winminheight = s:save_winminheight
let &winminwidth = s:save_winminwidth
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
set hlsearch
nohlsearch
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
