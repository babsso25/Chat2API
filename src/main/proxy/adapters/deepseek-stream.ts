/**
 * DeepSeek Stream Response Handler
 * Converts DeepSeek SSE stream to OpenAI compatible format
 */

import { PassThrough } from 'stream'
import { parseToolCallsFromText } from '../utils/toolParser'
import { 
  createToolCallState, 
  processStreamContent, 
  flushToolCallBuffer,
  createBaseChunk,
  ToolCallState 
} from '../utils/streamToolHandler'

const MODEL_NAME = 'deepseek-chat'

interface StreamChunk {
  p?: string
  v?: any
  response_message_id?: string
  o?: string
}

export class DeepSeekStreamHandler {
  private model: string
  private sessionId: string
  private isFirstChunk: boolean = true
  private messageId: string = ''
  private currentPath: string = ''
  private searchResults: any[] = []
  private thinkingStarted: boolean = false
  private accumulatedTokenUsage: number = 2
  private created: number
  private onEnd?: () => void
  private toolCallState: ToolCallState

  constructor(model: string, sessionId: string, onEnd?: () => void) {
    this.model = model
    this.sessionId = sessionId
    this.created = Math.floor(Date.now() / 1000)
    this.onEnd = onEnd
    this.toolCallState = createToolCallState()
  }

  private parseSSE(data: string): StreamChunk | null {
    try {
      return JSON.parse(data)
    } catch {
      return null
    }
  }

  private createChunk(delta: { role?: string; content?: string; reasoning_content?: string; tool_calls?: any[] }, finishReason?: string): string {
    return `data: ${JSON.stringify({
      id: `${this.sessionId}@${this.messageId}`,
      model: this.model,
      object: 'chat.completion.chunk',
      choices: [{
        index: 0,
        delta,
        finish_reason: finishReason || null,
      }],
      created: this.created,
    })}\n\n`
  }

  async handleStream(stream: NodeJS.ReadableStream): Promise<NodeJS.ReadableStream> {
    const transStream = new PassThrough()
    const isThinkingModel = this.model.includes('think') || this.model.includes('r1')
    const isSilentModel = this.model.includes('silent')
    const isFoldModel = (this.model.includes('fold') || this.model.includes('search')) && !isThinkingModel
    const isSearchSilentModel = this.model.includes('search-silent')

    let buffer = ''

    stream.on('data', (chunk: Buffer) => {
      buffer += chunk.toString()
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (!line.trim() || !line.startsWith('data:')) continue

        const data = line.slice(5).trim()
        if (data === '[DONE]') {
          this.handleDone(transStream, isFoldModel, isSearchSilentModel)
          return
        }

        const parsed = this.parseSSE(data)
        if (!parsed) continue

        this.processChunk(parsed, transStream, isThinkingModel, isSilentModel, isFoldModel, isSearchSilentModel)
      }
    })

    stream.on('end', () => {
      this.handleDone(transStream, isFoldModel, isSearchSilentModel)
    })

    stream.on('error', (err) => {
      transStream.emit('error', err)
    })

    return transStream
  }

  private processChunk(
    chunk: StreamChunk,
    transStream: PassThrough,
    isThinkingModel: boolean,
    isSilentModel: boolean,
    isFoldModel: boolean,
    isSearchSilentModel: boolean
  ): void {
    if (chunk.response_message_id && !this.messageId) {
      this.messageId = chunk.response_message_id
    }

    const previousPath = this.currentPath

    if (chunk.v && typeof chunk.v === 'object' && chunk.v.response) {
      const isThinkingNow = chunk.v.response.thinking_enabled
      this.currentPath = isThinkingNow ? 'thinking' : 'content'
      
      const fragments = chunk.v.response.fragments
      if (Array.isArray(fragments) && fragments.length > 0) {
        for (const fragment of fragments) {
          if (fragment.content) {
            const fragmentType = fragment.type
            const fragmentContent = fragment.content
            
            if (fragmentType === 'THINK') {
              this.sendContent(fragmentContent, 'thinking', transStream, isSilentModel, isFoldModel, isSearchSilentModel)
            } else if (fragmentType === 'ANSWER' || fragmentType === 'RESPONSE') {
              this.sendContent(fragmentContent, 'content', transStream, isSilentModel, isFoldModel, isSearchSilentModel)
            }
          }
        }
      }
    } else if (chunk.p === 'response/fragments') {
      this.currentPath = 'content'
      
      if (Array.isArray(chunk.v)) {
        for (const fragment of chunk.v) {
          if (fragment.content) {
            const fragmentType = fragment.type
            const fragmentContent = fragment.content
            
            if (fragmentType === 'THINK') {
              this.sendContent(fragmentContent, 'thinking', transStream, isSilentModel, isFoldModel, isSearchSilentModel)
            } else if (fragmentType === 'ANSWER' || fragmentType === 'RESPONSE') {
              this.sendContent(fragmentContent, 'content', transStream, isSilentModel, isFoldModel, isSearchSilentModel)
            }
          }
        }
      }
    } else if (chunk.p === 'response' && Array.isArray(chunk.v)) {
      const hasThinking = chunk.v.some((e: any) => 
        e.p === 'response' && e.v && typeof e.v === 'object' && e.v.thinking_enabled === true
      )
      if (hasThinking) {
        this.currentPath = 'thinking'
      }
    }

    if (chunk.p === 'response/search_status') return

    if (chunk.p === 'response' && Array.isArray(chunk.v)) {
      chunk.v.forEach((e: any) => {
        if (e.p === 'accumulated_token_usage' && typeof e.v === 'number') {
          this.accumulatedTokenUsage = e.v
        }
      })
    }

    if (chunk.p === 'response/search_results' && Array.isArray(chunk.v)) {
      if (chunk.o !== 'BATCH') {
        this.searchResults = chunk.v
      } else {
        chunk.v.forEach((op: any) => {
          const match = op.p?.match(/^(\d+)\/cite_index$/)
          if (match) {
            const index = parseInt(match[1], 10)
            if (this.searchResults[index]) {
              this.searchResults[index].cite_index = op.v
            }
          }
        })
      }
      return
    }

    let content = ''
    if (typeof chunk.v === 'string') {
      content = chunk.v
    } else if (Array.isArray(chunk.v)) {
      content = chunk.v
        .map((e: any) => {
          if (Array.isArray(e.v)) {
            return e.v.map((v: any) => v.content).join('')
          }
          return ''
        })
        .join('')
    }

    if (!content) return

    this.sendContent(content, this.currentPath, transStream, isSilentModel, isFoldModel, isSearchSilentModel)
  }

  private sendContent(
    content: string,
    path: string,
    transStream: PassThrough,
    isSilentModel: boolean,
    isFoldModel: boolean,
    isSearchSilentModel: boolean
  ): void {
    const cleanedValue = content.replace(/FINISHED/g, '')
    const processedContent = isSearchSilentModel
      ? cleanedValue.replace(/\[citation:(\d+)\]/g, '')
      : cleanedValue.replace(/\[citation:(\d+)\]/g, '[$1]')

    const delta: { role?: string; content?: string; reasoning_content?: string } = {}
    let shouldSendDelta = true

    if (this.isFirstChunk) {
      delta.role = 'assistant'
    }

    if (path === 'thinking') {
      if (isSilentModel) return

      const filteredContent = processedContent.replace(/^(SEARCH|WEB_SEARCH|SEARCHING)\s*/i, '')

      if (isFoldModel) {
        if (!this.thinkingStarted) {
          this.thinkingStarted = true
          delta.content = `<details><summary>Thinking Process</summary><pre>${filteredContent}`
        } else {
          delta.content = filteredContent
        }
      } else {
        if (filteredContent) {
          delta.reasoning_content = filteredContent
        } else {
          shouldSendDelta = false
        }
      }
    } else if (path === 'content') {
      if (isFoldModel && this.thinkingStarted) {
        delta.content = `</pre></details>${processedContent}`
        this.thinkingStarted = false
      } else {
        delta.content = processedContent
      }
    } else {
      delta.content = processedContent
    }

    if (shouldSendDelta && (delta.content !== undefined || delta.reasoning_content !== undefined)) {
      transStream.write(this.createChunk(delta))
      this.isFirstChunk = false
    }
  }

  private handleDone(transStream: PassThrough, isFoldModel: boolean, isSearchSilentModel: boolean): void {
    // Flush tool call buffer before finishing
    const baseChunk = createBaseChunk(`${this.sessionId}@${this.messageId}`, this.model, this.created)
    const flushChunks = flushToolCallBuffer(this.toolCallState, baseChunk, 'deepseek')
    for (const outChunk of flushChunks) {
      transStream.write(`data: ${JSON.stringify(outChunk)}\n\n`)
    }

    if (isFoldModel && this.thinkingStarted) {
      transStream.write(this.createChunk({ content: '</pre></details>' }))
    }

    if (this.searchResults.length > 0 && !isSearchSilentModel) {
      const citations = this.searchResults
        .filter(r => r.cite_index)
        .sort((a, b) => a.cite_index - b.cite_index)
        .map(r => `[${r.cite_index}]: [${r.title}](${r.url})`)
        .join('\n')
      
      if (citations) {
        transStream.write(this.createChunk({ content: `\n\n${citations}` }))
      }
    }

    // Determine finish_reason based on whether we had tool calls
    const finishReason = this.toolCallState.hasEmittedToolCall ? 'tool_calls' : 'stop'

    transStream.write(this.createChunk({}, finishReason))
    transStream.write('data: [DONE]\n\n')
    transStream.end()
    
    // Call end callback
    this.onEnd?.()
  }

  async handleNonStream(stream: NodeJS.ReadableStream): Promise<any> {
    let accumulatedContent = ''
    let accumulatedThinkingContent = ''
    let messageId = ''
    let currentPath = ''
    let accumulatedTokenUsage = 2

    return new Promise((resolve, reject) => {
      let buffer = ''

      stream.on('data', (chunk: Buffer) => {
        buffer += chunk.toString()
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.trim() || !line.startsWith('data:')) continue

          const data = line.slice(5).trim()
          if (data === '[DONE]') return

          try {
            const parsed = JSON.parse(data)
            
            if (parsed.response_message_id && !messageId) {
              messageId = parsed.response_message_id
            }

            if (parsed.v && typeof parsed.v === 'object' && parsed.v.response) {
              currentPath = parsed.v.response.thinking_enabled ? 'thinking' : 'content'
              
              const fragments = parsed.v.response.fragments
              if (Array.isArray(fragments) && fragments.length > 0) {
                for (const fragment of fragments) {
                  if (fragment.content) {
                    let cleanedFragment = fragment.content.replace(/FINISHED/g, '')
                    if (fragment.type === 'THINK') {
                      cleanedFragment = cleanedFragment.replace(/^(SEARCH|WEB_SEARCH|SEARCHING)\s*/i, '')
                      accumulatedThinkingContent += cleanedFragment
                    } else if (fragment.type === 'ANSWER' || fragment.type === 'RESPONSE') {
                      accumulatedContent += cleanedFragment
                    }
                  }
                }
              }
            } else if (parsed.p === 'response/fragments') {
              currentPath = 'content'
              
              if (Array.isArray(parsed.v)) {
                for (const fragment of parsed.v) {
                  if (fragment.content) {
                    let cleanedFragment = fragment.content.replace(/FINISHED/g, '')
                    if (fragment.type === 'THINK') {
                      cleanedFragment = cleanedFragment.replace(/^(SEARCH|WEB_SEARCH|SEARCHING)\s*/i, '')
                      accumulatedThinkingContent += cleanedFragment
                    } else if (fragment.type === 'ANSWER' || fragment.type === 'RESPONSE') {
                      accumulatedContent += cleanedFragment
                    }
                  }
                }
              }
            } else if (parsed.p === 'response' && Array.isArray(parsed.v)) {
              const hasThinking = parsed.v.some((e: any) => 
                e.p === 'response' && e.v && typeof e.v === 'object' && e.v.thinking_enabled === true
              )
              if (hasThinking) {
                currentPath = 'thinking'
              }
            }

            if (!currentPath) {
              // Default to thinking for thinking models
              if (parsed.v && typeof parsed.v === 'string') {
                currentPath = 'thinking'
              }
            }

            if (typeof parsed.v === 'object' && Array.isArray(parsed.v)) {
              parsed.v.forEach((e: any) => {
                if (e.accumulated_token_usage && typeof e.v === 'number') {
                  accumulatedTokenUsage = e.v
                }
                if (Array.isArray(e.v)) {
                  let cleanedValue = e.v.map((v: any) => v.content).join('').replace(/FINISHED/g, '')
                  if (currentPath === 'thinking') {
                    cleanedValue = cleanedValue.replace(/^(SEARCH|WEB_SEARCH|SEARCHING)\s*/i, '')
                    accumulatedThinkingContent += cleanedValue
                  } else if (currentPath === 'content') {
                    accumulatedContent += cleanedValue
                  }
                }
              })
            }

            if (typeof parsed.v === 'string') {
              let cleanedValue = parsed.v.replace(/FINISHED/g, '')
              if (currentPath === 'thinking') {
                cleanedValue = cleanedValue.replace(/^(SEARCH|WEB_SEARCH|SEARCHING)\s*/i, '')
                accumulatedThinkingContent += cleanedValue
              } else if (currentPath === 'content') {
                accumulatedContent += cleanedValue
              }
            }
          } catch {
            // Ignore parse errors
          }
        }
      })

      stream.on('end', () => {
        // Parse tool calls from accumulated content
        const { content: cleanContent, toolCalls } = parseToolCallsFromText(accumulatedContent)

        const message: any = {
          role: 'assistant',
          content: toolCalls.length > 0 ? null : cleanContent.trim(),
          reasoning_content: accumulatedThinkingContent.trim(),
        }

        if (toolCalls.length > 0) {
          message.tool_calls = toolCalls
        }

        resolve({
          id: `${this.sessionId}@${messageId}`,
          model: this.model,
          object: 'chat.completion',
          choices: [{
            index: 0,
            message,
            finish_reason: toolCalls.length > 0 ? 'tool_calls' : 'stop',
          }],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: accumulatedTokenUsage },
          created: this.created,
        })
      })

      stream.on('error', reject)
    })
  }
}
